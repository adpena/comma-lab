#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM OR1 scorer-free representation screen on the exact D3 merged n600 field.

This is a SCREEN, not a reference-form archive claim.  It retains and independently
decodes three real payloads:

* the shipped dense four-symbol field under deterministic RAW-LZMA2;
* the frame-to-frame XOR residual field under the same real coder; and
* row-boundary starts coded by the real SP1 contour/support coder.  The first pixel
  of every row is a start; subsequent starts mark class transitions.  Decoding the
  support and the class at each start reconstructs every dense row exactly.

The contour candidate is chunked and checkpointed.  A rerun resumes at the first
missing or invalid chunk.  All encoder streams and the final framed packet remain on
the SSD.  No scorer, archive mutation, or upstream write is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from measure_contour_string_flip_coding import (
    contour_decode_frames,
    contour_encode_frames,
)

from tac.boundary_math.dense_raster_lzma_baseline import (
    DenseRasterLzmaCode,
    decode_partition,
    encode_partition,
)

SCHEMA = "ddm_or1_orthogonal_representation_sweep.v1"
AXIS = "[macOS-CPU advisory / scorer-free exact rate and receiver screen, n600]"
EXPECTED_SOURCE_SHA256 = "deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07"
DEFAULT_SOURCE = Path(
    "/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/fields/"
    "tokens_lane_to_road_canonical.u8"
)
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep")
SHAPE = (600, 384, 512)
STREAM_ORDER = ("counts", "anchor", "chain", "cls")
PACKET_MAGIC = b"OR1C1\0\0\0"
RATE_DENOMINATOR = 37_545_489
GB1_TOKEN_STREAM_BYTES = 113_624
D3_FOUR_SYMBOL_STREAM_BYTES = 49_696


def _sha256_path(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_path(path)}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_bytes(path, data)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _load_dense4(source: Path) -> np.memmap:
    expected_bytes = int(np.prod(SHAPE))
    if source.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"source size mismatch: {source.stat().st_size} != {expected_bytes} bytes"
        )
    observed_sha = _sha256_path(source)
    if observed_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source sha mismatch: {observed_sha} != {EXPECTED_SOURCE_SHA256}")
    canonical = np.memmap(source, mode="r", dtype=np.uint8, shape=SHAPE)
    unique = np.unique(canonical)
    if unique.tolist() != [0, 2, 3, 4]:
        raise RuntimeError(f"expected canonical classes [0,2,3,4], got {unique.tolist()}")
    return canonical


def _canonical_to_dense4(canonical: np.ndarray) -> np.ndarray:
    # Lane was merged into Road by D3.  Preserve the four remaining symbols densely.
    lut = np.asarray([0, 255, 1, 2, 3], dtype=np.uint8)
    dense = lut[np.asarray(canonical, dtype=np.uint8)]
    if int(dense.max()) > 3:
        raise RuntimeError("unexpected symbol survived canonical-to-dense4 remap")
    return dense


def _dense_lzma_candidate(
    *, candidate_id: str, dense4: np.memmap, out_dir: Path, temporal: bool
) -> dict[str, Any]:
    payload_path = out_dir / "retained" / "payloads" / f"{candidate_id}.raw_lzma2"
    t0 = time.monotonic()
    if payload_path.exists():
        payload = payload_path.read_bytes()
        resumed = True
    else:
        if temporal:
            field = np.empty(SHAPE, dtype=np.uint8)
            field[0] = _canonical_to_dense4(dense4[0])
            for frame in range(1, SHAPE[0]):
                field[frame] = np.bitwise_xor(
                    _canonical_to_dense4(dense4[frame]),
                    _canonical_to_dense4(dense4[frame - 1]),
                )
        else:
            field = _canonical_to_dense4(dense4)
        code = encode_partition(field.reshape(SHAPE[0] * SHAPE[1], SHAPE[2]), n_classes=4)
        payload = code.payload
        _atomic_write_bytes(payload_path, payload)
        resumed = False

    code = DenseRasterLzmaCode(
        payload=payload, shape=(SHAPE[0] * SHAPE[1], SHAPE[2]), n_classes=4
    )
    decoded = decode_partition(code).astype(np.uint8).reshape(SHAPE)
    if temporal:
        for frame in range(1, SHAPE[0]):
            decoded[frame] = np.bitwise_xor(decoded[frame], decoded[frame - 1])
    for frame in range(SHAPE[0]):
        if not np.array_equal(decoded[frame], _canonical_to_dense4(dense4[frame])):
            raise RuntimeError(f"{candidate_id} decode mismatch at frame {frame}")
    return {
        "candidate_id": candidate_id,
        "mechanism": (
            "dense frame-to-frame XOR field, deterministic RAW-LZMA2"
            if temporal
            else "dense four-symbol raster, deterministic RAW-LZMA2"
        ),
        "payload": _artifact(payload_path),
        "receiver_closed": True,
        "decode_identity": True,
        "reference_form": False,
        "screen_scope": "standalone field packet; not integrated with shipped F26/HPAC runtime",
        "resumed": resumed,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }


def _boundary_starts(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense = _canonical_to_dense4(frame)
    starts = np.empty(dense.shape, dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = dense[:, 1:] != dense[:, :-1]
    classes = np.where(starts, dense, 0).astype(np.int64)
    return starts, classes


def _reconstruct_from_starts(starts: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if not bool(starts[:, 0].all()):
        raise RuntimeError("boundary packet omitted a row's first symbol")
    x = np.broadcast_to(np.arange(starts.shape[1], dtype=np.int64), starts.shape)
    last_start = np.maximum.accumulate(np.where(starts, x, 0), axis=1)
    return np.take_along_axis(classes, last_start, axis=1).astype(np.uint8)


def _encode_boundary_chunk(
    dense4: np.memmap, chunk_start: int, chunk_stop: int, chunk_dir: Path
) -> dict[str, Any]:
    chunk_id = f"frames_{chunk_start:04d}_{chunk_stop:04d}"
    receipt_path = chunk_dir / f"{chunk_id}.json"
    stream_paths = {name: chunk_dir / f"{chunk_id}.{name}.rc64" for name in STREAM_ORDER}
    if receipt_path.exists() and all(path.exists() for path in stream_paths.values()):
        receipt = json.loads(receipt_path.read_text())
        for name, path in stream_paths.items():
            if _artifact(path) != receipt["streams"][name]:
                raise RuntimeError(f"resume artifact drift: {path}")
        return receipt

    flip_maps: list[np.ndarray] = []
    class_maps: list[np.ndarray] = []
    for frame in range(chunk_start, chunk_stop):
        starts, classes = _boundary_starts(dense4[frame])
        flip_maps.append(starts)
        class_maps.append(classes)

    t0 = time.monotonic()
    encoded = contour_encode_frames(flip_maps, class_maps)
    streams = encoded.pop("streams")
    for name in STREAM_ORDER:
        _atomic_write_bytes(stream_paths[name], streams[name])

    decoded_starts, decoded_classes = contour_decode_frames(
        streams, n_frames=chunk_stop - chunk_start, h=SHAPE[1], w=SHAPE[2]
    )
    for local, frame in enumerate(range(chunk_start, chunk_stop)):
        reconstructed = _reconstruct_from_starts(decoded_starts[local], decoded_classes[local])
        if not np.array_equal(reconstructed, _canonical_to_dense4(dense4[frame])):
            raise RuntimeError(f"boundary-support decode mismatch at frame {frame}")

    receipt = {
        "schema": "ddm_or1_boundary_chunk.v1",
        "chunk_id": chunk_id,
        "frames": [chunk_start, chunk_stop],
        "mechanism": "SP1 contour/support coder over row class-transition starts",
        "streams": {name: _artifact(stream_paths[name]) for name in STREAM_ORDER},
        "stream_total_bytes": sum(path.stat().st_size for path in stream_paths.values()),
        "support_symbols": int(sum(int(fm.sum()) for fm in flip_maps)),
        "receiver_closed": True,
        "decode_identity": True,
        "coder_report": encoded,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _assemble_boundary_packet(receipts: list[dict[str, Any]], packet_path: Path) -> None:
    header = struct.pack(
        "<8sIIIII", PACKET_MAGIC, SHAPE[0], SHAPE[1], SHAPE[2], len(receipts), len(STREAM_ORDER)
    )
    packet = bytearray(header)
    for receipt in receipts:
        packet.extend(struct.pack("<II", *receipt["frames"]))
        for name in STREAM_ORDER:
            data = Path(receipt["streams"][name]["path"]).read_bytes()
            packet.extend(struct.pack("<Q", len(data)))
            packet.extend(data)
    _atomic_write_bytes(packet_path, bytes(packet))


def _decode_boundary_packet(packet_path: Path, dense4: np.memmap) -> None:
    packet = memoryview(packet_path.read_bytes())
    offset = 0
    header_size = struct.calcsize("<8sIIIII")
    magic, n_frames, height, width, n_chunks, n_streams = struct.unpack_from(
        "<8sIIIII", packet, offset
    )
    offset += header_size
    if (magic, n_frames, height, width, n_streams) != (
        PACKET_MAGIC,
        SHAPE[0],
        SHAPE[1],
        SHAPE[2],
        len(STREAM_ORDER),
    ):
        raise RuntimeError("boundary packet header mismatch")
    expected_start = 0
    for _ in range(n_chunks):
        chunk_start, chunk_stop = struct.unpack_from("<II", packet, offset)
        offset += struct.calcsize("<II")
        if chunk_start != expected_start or not chunk_start < chunk_stop:
            raise RuntimeError("boundary packet chunk sequence mismatch")
        streams: dict[str, bytes] = {}
        for name in STREAM_ORDER:
            (length,) = struct.unpack_from("<Q", packet, offset)
            offset += struct.calcsize("<Q")
            streams[name] = bytes(packet[offset : offset + length])
            offset += length
        starts, classes = contour_decode_frames(
            streams, n_frames=chunk_stop - chunk_start, h=height, w=width
        )
        for local, frame in enumerate(range(chunk_start, chunk_stop)):
            reconstructed = _reconstruct_from_starts(starts[local], classes[local])
            if not np.array_equal(reconstructed, _canonical_to_dense4(dense4[frame])):
                raise RuntimeError(f"packet parse-back mismatch at frame {frame}")
        expected_start = chunk_stop
    if expected_start != n_frames or offset != len(packet):
        raise RuntimeError("boundary packet trailing or missing bytes")


def _boundary_candidate(
    dense4: np.memmap, out_dir: Path, chunk_frames: int
) -> dict[str, Any]:
    chunk_dir = out_dir / "retained" / "boundary_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    t0 = time.monotonic()
    for start in range(0, SHAPE[0], chunk_frames):
        stop = min(SHAPE[0], start + chunk_frames)
        receipt = _encode_boundary_chunk(dense4, start, stop, chunk_dir)
        receipts.append(receipt)
        print(
            f"[ddm_or1] boundary {start:04d}:{stop:04d} "
            f"{receipt['stream_total_bytes']:,} B",
            flush=True,
        )

    packet_path = out_dir / "retained" / "payloads" / "boundary_starts_sp1.packet"
    _assemble_boundary_packet(receipts, packet_path)
    _decode_boundary_packet(packet_path, dense4)
    return {
        "candidate_id": "boundary_starts_sp1",
        "mechanism": "framed SP1 contour/support streams over row class-transition starts",
        "payload": _artifact(packet_path),
        "payload_stream_bytes": sum(r["stream_total_bytes"] for r in receipts),
        "packet_framing_bytes": packet_path.stat().st_size
        - sum(r["stream_total_bytes"] for r in receipts),
        "support_symbols": sum(r["support_symbols"] for r in receipts),
        "chunk_frames": chunk_frames,
        "chunks": len(receipts),
        "chunk_receipts": [
            str(chunk_dir / f"frames_{r['frames'][0]:04d}_{r['frames'][1]:04d}.json")
            for r in receipts
        ],
        "receiver_closed": True,
        "decode_identity": True,
        "reference_form": False,
        "screen_scope": "standalone boundary packet; not integrated with shipped F26/HPAC runtime",
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }


def _price_row(row: dict[str, Any]) -> dict[str, Any]:
    size = int(row["payload"]["bytes"])
    row = dict(row)
    row["delta_vs_gb1_token_stream_bytes"] = size - GB1_TOKEN_STREAM_BYTES
    row["delta_vs_d3_four_symbol_stream_bytes"] = size - D3_FOUR_SYMBOL_STREAM_BYTES
    row["delta_S_rate_vs_gb1_token_stream"] = 25 * (
        size - GB1_TOKEN_STREAM_BYTES
    ) / RATE_DENOMINATOR
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument("--minimum-free-bytes", type=int, default=2 << 30)
    parser.add_argument(
        "--determinism-repeat",
        action="store_true",
        help="encode every candidate again into a distinct retained tree and require identical bytes",
    )
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.out_dir).free
    if free_bytes < args.minimum_free_bytes:
        raise RuntimeError(
            f"storage preflight failed: {free_bytes} < {args.minimum_free_bytes} free bytes"
        )

    started = datetime.now(UTC)
    dense4 = _load_dense4(args.source)
    candidates = [
        _dense_lzma_candidate(
            candidate_id="dense_raster_raw_lzma2",
            dense4=dense4,
            out_dir=args.out_dir,
            temporal=False,
        ),
        _dense_lzma_candidate(
            candidate_id="temporal_xor_raw_lzma2",
            dense4=dense4,
            out_dir=args.out_dir,
            temporal=True,
        ),
        _boundary_candidate(dense4, args.out_dir, args.chunk_frames),
    ]
    candidates = [_price_row(row) for row in candidates]
    determinism_repeat: dict[str, Any] | None = None
    if args.determinism_repeat:
        repeat_dir = (
            args.out_dir
            / "determinism_repeats"
            / started.strftime("%Y%m%dT%H%M%S%fZ")
        )
        if repeat_dir.exists():
            raise RuntimeError(f"fresh determinism-repeat path already exists: {repeat_dir}")
        repeat_candidates = [
            _dense_lzma_candidate(
                candidate_id="dense_raster_raw_lzma2",
                dense4=dense4,
                out_dir=repeat_dir,
                temporal=False,
            ),
            _dense_lzma_candidate(
                candidate_id="temporal_xor_raw_lzma2",
                dense4=dense4,
                out_dir=repeat_dir,
                temporal=True,
            ),
            _boundary_candidate(dense4, repeat_dir, args.chunk_frames),
        ]
        first_by_id = {row["candidate_id"]: row for row in candidates}
        repeat_rows = []
        for repeat_row in repeat_candidates:
            first = first_by_id[repeat_row["candidate_id"]]
            identical = repeat_row["payload"]["sha256"] == first["payload"]["sha256"]
            if not identical or repeat_row["payload"]["bytes"] != first["payload"]["bytes"]:
                raise RuntimeError(f"determinism repeat mismatch: {repeat_row['candidate_id']}")
            repeat_rows.append(
                {
                    "candidate_id": repeat_row["candidate_id"],
                    "first": first["payload"],
                    "repeat": repeat_row["payload"],
                    "byte_identical": True,
                }
            )
        determinism_repeat = {
            "complete": True,
            "all_payloads_byte_identical": True,
            "rows": repeat_rows,
        }
    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "complete": True,
        "score_claim": False,
        "promotion_eligible": False,
        "reference_form": False,
        "source": _artifact(args.source),
        "source_shape": list(SHAPE),
        "source_classes": [0, 2, 3, 4],
        "dense_classes": [0, 1, 2, 3],
        "git_head_at_launch": _git_head(),
        "source_code": {
            "sweep": _artifact(Path(__file__).resolve()),
            "sp1_contour_coder": _artifact(
                _REPO / "tools" / "measure_contour_string_flip_coding.py"
            ),
            "dense_raster_coder": _artifact(
                _REPO / "src" / "tac" / "boundary_math" / "dense_raster_lzma_baseline.py"
            ),
        },
        "argv": sys.argv,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "storage": {
            "path": str(args.out_dir),
            "observed_free_bytes": free_bytes,
            "minimum_free_bytes": args.minimum_free_bytes,
            "status": "PASS",
        },
        "baselines": {
            "gb1_groupbin8_token_stream_bytes": GB1_TOKEN_STREAM_BYTES,
            "d3_four_symbol_f26_hpac_stream_bytes": D3_FOUR_SYMBOL_STREAM_BYTES,
            "rate_denominator_bytes": RATE_DENOMINATOR,
        },
        "candidates": candidates,
        "best_candidate_id": min(candidates, key=lambda row: row["payload"]["bytes"])[
            "candidate_id"
        ],
        "screen_boundary": (
            "Real retained bytes and independent exact parse-back, but standalone non-reference "
            "coders; only the D3 F26/HPAC baseline is receiver-integrated reference form."
        ),
    }
    if determinism_repeat is not None:
        result["determinism_repeat"] = determinism_repeat
    result_path = args.out_dir / "RESULT.json"
    _atomic_write_json(result_path, result)
    manifest = {
        "schema": "ddm_or1_orthogonal_payload_manifest.v1",
        "result": _artifact(result_path),
        "source": result["source"],
        "payloads": [row["payload"] for row in candidates],
        "repeat_payloads": (
            [row["repeat"] for row in determinism_repeat["rows"]]
            if determinism_repeat is not None
            else []
        ),
        "source_code": result["source_code"],
        "chunk_receipts": candidates[-1]["chunk_receipts"],
    }
    _atomic_write_json(args.out_dir / "PAYLOAD_MANIFEST.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
