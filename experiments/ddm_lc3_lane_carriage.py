#!/usr/bin/env python3
# ruff: noqa: I001
"""LC3: exact cropped-SMEVR Lane carriage over the D3 four-class quotient.

This scorer-free arm measures one previously unraced lossless representation:
the exact Lane mask is cropped only across globally empty rows, split into
bounded tiles, and encoded with the generic SMEVR temporal event/value coder.
The counted packet carries the crop bounds and every SMEVR frame.  Its receiver
has no access to the source mask: it parses the packet, restores zero rows,
paints Lane over the receiver-closed D3 quotient, and must reproduce the exact
source token-field SHA-256.

Every materialized payload is retained under ``--resume-from`` with bytes and
SHA-256.  Build and receive are separate resumable stages.  The cleanup stage
removes only certified partial files after a complete exact receiver receipt;
candidate payloads and stage checkpoints are always preserved.

AXIS: [scorer-free exact rate and receiver measurement].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_d3_alphabet_merge as d3
from experiments import ddm_r7_token_coder as r7


STORE = Path("/Volumes/APDataStore/pact/ddm_lc3")
D3_STORE = Path("/Volumes/APDataStore/pact/ddm_d3_alphabet_merge")
N, H, W = 600, 384, 512
FIELD_BYTES = N * H * W
SOURCE_FIELD_SHA256 = d3.SOURCE_FIELD_SHA256
D3_MERGED_SHA256 = "deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07"
LANE_MASK_SHA256 = "6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf"
PACKET_MAGIC = b"LC3S"
PACKET_VERSION = 1
PACKET_HEADER = struct.Struct("<4sB6H")
TILE_HEADER = struct.Struct("<HI")
TILE_ROWS = 48
MINIMUM_FREE_BYTES = 2 << 30
LANE_CARRIAGE_BAR_BYTES = 21_699
GF1_INCUMBENT_BYTES = 36_044
RATE_SCORE_PER_BYTE = 6.658589531221714e-7
RATE_ONLY_ARCHIVE_BYTES = 116_287
AXIS = "[scorer-free exact rate and receiver measurement]"


class LC3Error(RuntimeError):
    """A custody, retention, framing, or exact-receiver gate refused."""


def validate_store(store: Path) -> Path:
    resolved = store.resolve()
    if resolved != STORE.resolve():
        raise LC3Error(f"LC3 custody is pinned to {STORE}, not {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def progress(**record: Any) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def require_fact(path: Path, *, size: int, sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != sha256:
        raise LC3Error(f"{label} custody drifted: {path}")
    return file_fact(path)


def preflight(store: Path) -> dict[str, Any]:
    store = validate_store(store)
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise LC3Error(f"storage preflight failed at {store}: {free} B free < {MINIMUM_FREE_BYTES} B")
    proof_path = D3_STORE / "DECODE_RESULT.json"
    if not proof_path.is_file():
        raise LC3Error("D3 independent receiver receipt is absent")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if not (proof.get("complete") and proof.get("receiver_closed") and proof.get("byte_identical")):
        raise LC3Error("D3 independent receiver receipt is not complete and byte-identical")
    facts = {
        "schema": "ddm_lc3_preflight.v1",
        "complete": True,
        "storage": {
            "path": str(store),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "observed_free_bytes": free,
            "status": "PASS",
        },
        "source_field": require_fact(
            d3.SOURCE_FIELD,
            size=FIELD_BYTES,
            sha256=SOURCE_FIELD_SHA256,
            label="exact source token field",
        ),
        "d3_merged_field": require_fact(
            d3.retained_paths(D3_STORE)["merged"],
            size=FIELD_BYTES,
            sha256=D3_MERGED_SHA256,
            label="D3 receiver-closed quotient field",
        ),
        "d3_exact_lane_mask": require_fact(
            d3.retained_paths(D3_STORE)["class1_mask"],
            size=(FIELD_BYTES + 7) // 8,
            sha256=LANE_MASK_SHA256,
            label="D3 exact Lane mask",
        ),
        "d3_receiver_receipt": file_fact(proof_path),
        "seed": None,
        "determinism": "integer-only canonical codec; no RNG",
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "checkpoints/STAGE0_PREFLIGHT.json", facts)
    return facts


def load_exact_mask() -> np.ndarray:
    path = d3.retained_paths(D3_STORE)["class1_mask"]
    packed = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[:FIELD_BYTES]
    return bits.reshape(N, H, W).astype(np.uint8, copy=False)


def crop_bounds(mask: np.ndarray) -> tuple[int, int]:
    occupied = np.flatnonzero(np.any(mask != 0, axis=(0, 2)))
    if occupied.size == 0:
        return 0, 0
    return int(occupied[0]), int(occupied[-1]) + 1


def tile_fact(path: Path, row_start: int, row_stop: int) -> dict[str, Any]:
    fact = file_fact(path)
    accounting = r7.frame_accounting(path.read_bytes())
    return {
        **fact,
        "row_start": row_start,
        "row_stop_exclusive": row_stop,
        "codec": accounting.codec,
        "header_bytes": accounting.header_bytes,
        "base_bytes": accounting.base_bytes,
        "delta_bytes": accounting.delta_bytes,
        "raw_token_bytes": accounting.raw_token_bytes,
    }


def build_packet(store: Path) -> dict[str, Any]:
    custody = preflight(store)
    root = store / "retained/cropped_smevr"
    root.mkdir(parents=True, exist_ok=True)
    mask = load_exact_mask()
    row_start, row_stop = crop_bounds(mask)
    if row_start >= row_stop:
        raise LC3Error("exact Lane mask is unexpectedly empty")
    if np.any(mask[:, :row_start]) or np.any(mask[:, row_stop:]):
        raise LC3Error("derived crop would discard a Lane pixel")
    ranges = [(start, min(start + TILE_ROWS, row_stop)) for start in range(row_start, row_stop, TILE_ROWS)]
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges):
        tile_path = root / f"tile_{index:02d}_rows_{start}_{stop}.r7"
        if tile_path.is_file():
            frame = tile_path.read_bytes()
            restored = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
            expected = np.ascontiguousarray(mask[:, start:stop, :, None])
            if not np.array_equal(restored, expected):
                raise LC3Error(f"retained SMEVR tile {index} no longer decodes exactly")
        else:
            source = np.ascontiguousarray(mask[:, start:stop, :, None])
            frame = r7.encode_token_codes(source, levels=2, codec="smevr")
            atomic_bytes(tile_path, frame)
            restored = r7.decode_token_codes(frame, verify=r7.VERIFY_DIGEST)
            if not np.array_equal(restored, source):
                raise LC3Error(f"SMEVR tile {index} changed its input")
        completed.append(tile_fact(tile_path, start, stop))
        checkpoint = {
            "schema": "ddm_lc3_build_checkpoint.v1",
            "complete_tiles": completed,
            "planned_tile_count": len(ranges),
            "crop": {"row_start": row_start, "row_stop_exclusive": row_stop},
            "checkpoint_complete": True,
        }
        atomic_json(store / "checkpoints/STAGE1_BUILD_PROGRESS.json", checkpoint)
        progress(
            stage="build",
            event="tile_complete",
            tile=index + 1,
            tiles=len(ranges),
            bytes=tile_path.stat().st_size,
            cumulative_bytes=sum(int(item["bytes"]) for item in completed),
        )

    packet = bytearray(
        PACKET_HEADER.pack(
            PACKET_MAGIC,
            PACKET_VERSION,
            N,
            H,
            W,
            row_start,
            row_stop,
            len(completed),
        )
    )
    for item in completed:
        frame = Path(item["path"]).read_bytes()
        rows = int(item["row_stop_exclusive"]) - int(item["row_start"])
        packet.extend(TILE_HEADER.pack(rows, len(frame)))
        packet.extend(frame)
    packet_path = root / "lane_exact_cropped_smevr.lc3"
    atomic_bytes(packet_path, bytes(packet))
    packet_bytes = packet_path.stat().st_size
    result = {
        "schema": "ddm_lc3_cropped_smevr_build.v1",
        "complete": True,
        "mechanism": (
            "counted global nonzero-row bounds; exact binary Lane crop; canonical row tiles; "
            "R7 SMEVR temporal mode plus event/value residual"
        ),
        "source_mask": custody["d3_exact_lane_mask"],
        "source_lane_pixels": int(mask.sum()),
        "crop": {
            "row_start": row_start,
            "row_stop_exclusive": row_stop,
            "rows_counted": row_stop - row_start,
            "outside_crop_lane_pixels": 0,
        },
        "tile_rows": TILE_ROWS,
        "tiles": completed,
        "packet": file_fact(packet_path),
        "measured_carriage_bytes": packet_bytes,
        "arithmetic_floor_bytes": PACKET_HEADER.size + len(completed) * TILE_HEADER.size,
        "composed_archive_bytes_projection": RATE_ONLY_ARCHIVE_BYTES + packet_bytes,
        "projection_vs_sub012_bar_bytes": packet_bytes - LANE_CARRIAGE_BAR_BYTES,
        "projection_delta_S_vs_sub012_bar": (packet_bytes - LANE_CARRIAGE_BAR_BYTES) * RATE_SCORE_PER_BYTE,
        "delta_vs_gf1_incumbent_bytes": packet_bytes - GF1_INCUMBENT_BYTES,
        "identity_verdict": "PENDING_INDEPENDENT_RECEIVER",
        "retention": "ALL_SMEVR_TILES_AND_COMPOSED_PACKET",
        "elapsed_seconds": time.perf_counter() - started,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "checkpoints/STAGE1_BUILD_COMPLETE.json", result)
    return result


def parse_packet(packet: bytes, *, verify: str) -> tuple[np.ndarray, dict[str, Any]]:
    if len(packet) < PACKET_HEADER.size:
        raise LC3Error("LC3 packet is truncated before its header")
    magic, version, n, h, w, row_start, row_stop, tile_count = PACKET_HEADER.unpack_from(packet)
    if (magic, version, n, h, w) != (PACKET_MAGIC, PACKET_VERSION, N, H, W):
        raise LC3Error("LC3 packet identity or geometry differs")
    if not (0 <= row_start < row_stop <= H) or tile_count <= 0:
        raise LC3Error("LC3 crop bounds or tile count is invalid")
    output = np.zeros((N, H, W), dtype=np.uint8)
    offset = PACKET_HEADER.size
    cursor = row_start
    rows: list[dict[str, Any]] = []
    for index in range(tile_count):
        if len(packet) < offset + TILE_HEADER.size:
            raise LC3Error(f"LC3 tile {index} header is truncated")
        tile_rows, frame_length = TILE_HEADER.unpack_from(packet, offset)
        offset += TILE_HEADER.size
        stop = cursor + tile_rows
        if tile_rows <= 0 or stop > row_stop or len(packet) < offset + frame_length:
            raise LC3Error(f"LC3 tile {index} bounds or length is invalid")
        frame = packet[offset : offset + frame_length]
        offset += frame_length
        decoded = r7.decode_token_codes(frame, verify=verify)
        if decoded.shape != (N, tile_rows, W, 1) or decoded.dtype != np.uint8:
            raise LC3Error(f"LC3 tile {index} decoded geometry differs")
        output[:, cursor:stop] = decoded[..., 0]
        rows.append(
            {
                "tile": index,
                "row_start": cursor,
                "row_stop_exclusive": stop,
                "frame_bytes": frame_length,
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
            }
        )
        cursor = stop
    if cursor != row_stop or offset != len(packet):
        raise LC3Error("LC3 packet leaves a row gap or trailing bytes")
    return output, {
        "crop": {"row_start": row_start, "row_stop_exclusive": row_stop},
        "tiles": rows,
        "packet_bytes_consumed": offset,
        "verification": verify,
        "length_closed": True,
    }


def receive_packet(store: Path) -> dict[str, Any]:
    custody = preflight(store)
    build_path = store / "checkpoints/STAGE1_BUILD_COMPLETE.json"
    if not build_path.is_file():
        raise LC3Error("independent receiver requires the complete build receipt")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    packet_path = Path(build["packet"]["path"])
    if file_fact(packet_path) != build["packet"]:
        raise LC3Error("counted packet drifted since build")
    started = time.perf_counter()
    mask, parse = parse_packet(packet_path.read_bytes(), verify=r7.VERIFY_CANONICAL)
    mask_path = store / "retained/receiver/lane_mask_receiver.packbits"
    atomic_bytes(mask_path, np.packbits(mask.reshape(-1), bitorder="little").tobytes())

    merged_path = Path(custody["d3_merged_field"]["path"])
    merged = np.memmap(merged_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    output_path = store / "retained/receiver/source_tokens_receiver.u8"
    temporary = output_path.with_suffix(".u8.partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as output:
        for frame in range(N):
            plane = np.asarray(merged[frame], dtype=np.uint8).copy()
            lane = mask[frame] != 0
            if np.any(plane[lane] != 0):
                raise LC3Error(f"receiver frame {frame}: Lane packet paints outside quotient Road")
            plane[lane] = 1
            output.write(plane.tobytes(order="C"))
            if (frame + 1) % 100 == 0 or frame + 1 == N:
                atomic_json(
                    store / "checkpoints/STAGE2_RECEIVER_PROGRESS.json",
                    {
                        "schema": "ddm_lc3_receiver_checkpoint.v1",
                        "frame_stop_exclusive": frame + 1,
                        "partial_payload": {
                            "path": str(temporary),
                            "bytes": temporary.stat().st_size,
                        },
                        "checkpoint_complete": True,
                    },
                )
    os.replace(temporary, output_path)
    identity = sha256_file(output_path) == SOURCE_FIELD_SHA256
    source_mask = load_exact_mask()
    mask_identity = np.array_equal(mask, source_mask)
    if not identity or not mask_identity:
        raise LC3Error("independent receiver did not recover the exact Lane mask and token field")
    packet_bytes = packet_path.stat().st_size
    result = {
        "schema": "ddm_lc3_cropped_smevr_receiver.v1",
        "complete": True,
        "packet": file_fact(packet_path),
        "parsed": parse,
        "decoded_lane_mask": file_fact(mask_path),
        "decoded_token_field": file_fact(output_path),
        "expected_token_field_sha256": SOURCE_FIELD_SHA256,
        "lane_mask_byte_identical": True,
        "token_field_byte_identical": True,
        "identity_verdict": "EXACT",
        "measured_carriage_bytes": packet_bytes,
        "composed_archive_bytes_projection": RATE_ONLY_ARCHIVE_BYTES + packet_bytes,
        "projection_vs_sub012_bar_bytes": packet_bytes - LANE_CARRIAGE_BAR_BYTES,
        "projection_delta_S_vs_sub012_bar": (packet_bytes - LANE_CARRIAGE_BAR_BYTES) * RATE_SCORE_PER_BYTE,
        "delta_vs_gf1_incumbent_bytes": packet_bytes - GF1_INCUMBENT_BYTES,
        "elapsed_seconds": time.perf_counter() - started,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "checkpoints/STAGE2_RECEIVER_COMPLETE.json", result)
    return result


def summarize(store: Path) -> dict[str, Any]:
    store = validate_store(store)
    receiver_path = store / "checkpoints/STAGE2_RECEIVER_COMPLETE.json"
    if not receiver_path.is_file():
        raise LC3Error("summary requires the independent receiver receipt")
    receiver = json.loads(receiver_path.read_text(encoding="utf-8"))
    carriage = int(receiver["measured_carriage_bytes"])
    if carriage <= LANE_CARRIAGE_BAR_BYTES:
        verdict = "TAKEN"
        reason = "exact carriage meets the sub-0.12 byte bar"
    elif carriage < GF1_INCUMBENT_BYTES:
        verdict = "BUILT-RACED"
        reason = "exact carriage improves GF1 but remains above the sub-0.12 byte bar"
    else:
        verdict = "DEAD"
        reason = "exact carriage does not beat the GF1 incumbent"
    result = {
        "schema": "ddm_lc3_summary.v1",
        "complete": True,
        "candidate_id": "cropped_smevr_exact_lane",
        "mechanism": "counted vertical crop plus exact binary SMEVR row tiles",
        "arithmetic_floor_bytes": (PACKET_HEADER.size + len(receiver["parsed"]["tiles"]) * TILE_HEADER.size),
        "built": True,
        "measured_bytes": carriage,
        "identity_verdict": receiver["identity_verdict"],
        "composed_archive_bytes": RATE_ONLY_ARCHIVE_BYTES + carriage,
        "projection_vs_sub012_exchange": {
            "bar_bytes": LANE_CARRIAGE_BAR_BYTES,
            "delta_bytes": carriage - LANE_CARRIAGE_BAR_BYTES,
            "score_units_per_byte": RATE_SCORE_PER_BYTE,
            "delta_score_units": (carriage - LANE_CARRIAGE_BAR_BYTES) * RATE_SCORE_PER_BYTE,
            "projection_only": True,
        },
        "delta_vs_gf1_incumbent_bytes": carriage - GF1_INCUMBENT_BYTES,
        "verdict": verdict,
        "verdict_reason": reason,
        "joint_reencode": (
            "OWED_ON_SURVIVOR"
            if verdict in {"TAKEN", "BUILT-RACED"}
            else "NOT_FIRED; mechanism failed the carriage bar before archive composition"
        ),
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "SUMMARY.json", result)
    return result


def falsify_receiver(store: Path) -> dict[str, Any]:
    store = validate_store(store)
    receiver_path = store / "checkpoints/STAGE2_RECEIVER_COMPLETE.json"
    if not receiver_path.is_file():
        raise LC3Error("falsifier requires the complete exact receiver receipt")
    receiver = json.loads(receiver_path.read_text(encoding="utf-8"))
    packet_path = Path(receiver["packet"]["path"])
    if file_fact(packet_path) != receiver["packet"]:
        raise LC3Error("counted packet drifted before its falsifier")
    source = packet_path.read_bytes()
    corrupted = bytearray(source)
    corrupted[-1] ^= 1
    corrupt_path = store / "retained/falsifier/lane_exact_cropped_smevr_last_bit_flipped.lc3"
    atomic_bytes(corrupt_path, bytes(corrupted))
    refusal: str | None = None
    try:
        parse_packet(bytes(corrupted), verify=r7.VERIFY_CANONICAL)
    except (LC3Error, r7.DDMR7CoderError) as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    if refusal is None:
        raise LC3Error("one-bit corrupt packet was not refused by the independent receiver")
    result = {
        "schema": "ddm_lc3_receiver_falsifier.v1",
        "complete": True,
        "positive_control": {
            "packet": receiver["packet"],
            "token_field_byte_identical": receiver["token_field_byte_identical"],
            "decoded_token_field": receiver["decoded_token_field"],
            "verdict": "PASS",
        },
        "negative_control": {
            "mutation": "XOR 1 into the final counted packet byte",
            "payload": file_fact(corrupt_path),
            "receiver_verification": r7.VERIFY_CANONICAL,
            "receiver_refusal": refusal,
            "verdict": "PASS",
        },
        "instrument_verdict": "SENSITIVE_TO_ONE_BIT_PAYLOAD_CORRUPTION",
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "FALSIFIER_RESULT.json", result)
    return result


def cleanup(store: Path) -> dict[str, Any]:
    store = validate_store(store)
    receiver_path = store / "checkpoints/STAGE2_RECEIVER_COMPLETE.json"
    if not receiver_path.is_file():
        raise LC3Error("cleanup blocks until an exact receiver receipt exists")
    receiver = json.loads(receiver_path.read_text(encoding="utf-8"))
    if not (
        receiver.get("complete")
        and receiver.get("lane_mask_byte_identical")
        and receiver.get("token_field_byte_identical")
    ):
        raise LC3Error("cleanup blocks because the receiver receipt is not exact")
    removed: list[str] = []
    for path in sorted(store.rglob("*.partial")):
        if not path.is_file():
            continue
        path.unlink()
        removed.append(str(path))
    result = {
        "schema": "ddm_lc3_cleanup.v1",
        "complete": True,
        "scope": "certified incomplete partial files only",
        "removed": removed,
        "candidate_payloads_preserved": True,
        "receiver_payloads_preserved": True,
    }
    atomic_json(store / "CLEANUP_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "build", "receive", "summarize", "falsify", "cleanup", "all"),
    )
    parser.add_argument("--resume-from", default=str(STORE))
    args = parser.parse_args()
    store = validate_store(Path(args.resume_from))
    if args.stage == "preflight":
        result = preflight(store)
    elif args.stage == "build":
        result = build_packet(store)
    elif args.stage == "receive":
        result = receive_packet(store)
    elif args.stage == "summarize":
        result = summarize(store)
    elif args.stage == "falsify":
        result = falsify_receiver(store)
    elif args.stage == "cleanup":
        result = cleanup(store)
    else:
        build_packet(store)
        receive_packet(store)
        result = summarize(store)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
