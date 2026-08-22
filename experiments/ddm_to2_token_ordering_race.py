#!/usr/bin/env python3
"""Scorer-free, exact-invertible ordering race for the pinned DX2 token field."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import shutil
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import brotli
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1")
DX2_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/archive.zip")
DX2_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DX2_TOKEN_RECEIPT = DX2_TOKENS.with_suffix(".json")
AD2_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6/RESULT.json"
)
RB1_MEMO = ROOT / ".omx/research/ddm_rb1_rate_bound_decomposition_20260822.md"

EXPECTED = {
    DX2_ARCHIVE: (180_368, "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"),
    DX2_TOKENS: (117_964_800, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    DX2_TOKEN_RECEIPT: (
        None,
        "c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9",
    ),
    AD2_RESULT: (None, "80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511"),
    RB1_MEMO: (None, "fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09"),
}

T, H, W = 600, 384, 512
ALPHABET = 5
SITES = H * W
TOKEN_COUNT = T * SITES
INCUMBENT_TOKEN_BYTES = 113_777
SHIPPED_STREAM_DIGEST = "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"
INCUMBENT_ARCHIVE_BYTES = 180_368
INCUMBENT_SCORE = 0.14821987563243377
FIXED_DISTORTION = 0.0281202279752971
RATE_DENOMINATOR = 37_545_489
RATE_WEIGHT = 25
ARCHIVE_CEILING = 137_986
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_RESIDUAL_BYTES = 96
CLASS_HEADER = struct.Struct("<4sHHHBB5Q")
CODERS = ("brotli_q11", "lzma1_1m", "zlib9")
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 128,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_pin(path: Path, expected_bytes: int | None, expected_sha: str) -> dict[str, Any]:
    receipt = file_receipt(path)
    if expected_bytes is not None and receipt["bytes"] != expected_bytes:
        raise RuntimeError(f"pinned size drift for {path}: {receipt['bytes']} != {expected_bytes}")
    if receipt["sha256"] != expected_sha:
        raise RuntimeError(f"pinned sha drift for {path}: {receipt['sha256']} != {expected_sha}")
    return receipt


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = sha256_bytes(payload)
    if path.exists() and not replace:
        receipt = file_receipt(path)
        if receipt["bytes"] != len(payload) or receipt["sha256"] != expected_sha:
            raise RuntimeError(f"refusing to overwrite differing retained payload: {path}")
        return receipt
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)
    return {"path": str(path), "bytes": len(payload), "sha256": expected_sha}


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return atomic_bytes(path, payload, replace=replace)


def atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_receipt = file_receipt(source)
    if destination.exists():
        receipt = file_receipt(destination)
        if receipt["bytes"] != source_receipt["bytes"] or receipt["sha256"] != source_receipt["sha256"]:
            raise RuntimeError(f"refusing to overwrite differing retained input: {destination}")
        return receipt
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if file_receipt(temporary)["sha256"] != source_receipt["sha256"]:
        raise RuntimeError(f"copy verification failed: {source} -> {destination}")
    os.replace(temporary, destination)
    fsync_parent(destination)
    return file_receipt(destination)


def validate_receipt(receipt: dict[str, Any]) -> None:
    actual = file_receipt(Path(receipt["path"]))
    if actual["bytes"] != receipt["bytes"] or actual["sha256"] != receipt["sha256"]:
        raise RuntimeError(f"retained artifact drift: {receipt['path']}")


def cleanup_incomplete_temps(output: Path) -> list[dict[str, Any]]:
    """Certify and remove only this runner's atomic-write scratch remnants."""
    cleanup_path = output / "CLEANUP.json"
    prior: list[dict[str, Any]] = []
    if cleanup_path.exists():
        prior = json.loads(cleanup_path.read_text()).get("entries", [])
    targets = sorted(
        path
        for path in output.rglob(".*.tmp.*")
        if path.is_file() and path != cleanup_path
    )
    if not targets:
        return prior
    pending = [
        {
            **file_receipt(path),
            "reason": "incomplete atomic-write scratch; deterministically rebuildable from pinned input",
            "action": "delete",
            "status": "certified_pending",
        }
        for path in targets
    ]
    atomic_json(
        cleanup_path,
        {"schema": "ddm.to2.cleanup.v1", "entries": prior + pending},
        replace=True,
    )
    for path in targets:
        path.unlink()
        fsync_parent(path)
    completed = [dict(entry, status="deleted") for entry in pending]
    atomic_json(
        cleanup_path,
        {"schema": "ddm.to2.cleanup.v1", "entries": prior + completed},
        replace=True,
    )
    return prior + completed


def parse_anatomy(member: bytes, archive_bytes: int) -> tuple[dict[str, Any], bytes]:
    magic, version, codec, table_mode, feature_bits, hpac_len, semantic_len, carrier_len = (
        RX1_HEADER.unpack_from(member)
    )
    if magic != b"RX1M" or version != 1:
        raise RuntimeError("pinned DX2 RX1 header drifted")
    cursor = RX1_HEADER.size
    hpac = member[cursor : cursor + hpac_len]
    cursor += hpac_len
    semantic = member[cursor : cursor + semantic_len]
    cursor += semantic_len
    carrier = member[cursor : cursor + carrier_len]
    cursor += carrier_len
    residual = member[cursor : cursor + RX1_RESIDUAL_BYTES]
    cursor += RX1_RESIDUAL_BYTES
    tokens = member[cursor:]
    anatomy = {
        "archive_bytes": archive_bytes,
        "zip_framing_bytes": archive_bytes - len(member),
        "member_bytes": len(member),
        "rx1_header_bytes": RX1_HEADER.size,
        "hpac_bytes": len(hpac),
        "semantic_bytes": len(semantic),
        "carrier_bytes": len(carrier),
        "residual_bytes": len(residual),
        "token_stream_bytes": len(tokens),
        "header_fields": {
            "codec": codec,
            "table_mode": table_mode,
            "feature_bits": feature_bits,
        },
    }
    expected = {
        "zip_framing_bytes": 100,
        "rx1_header_bytes": 14,
        "hpac_bytes": 13_515,
        "semantic_bytes": 30_856,
        "carrier_bytes": 22_010,
        "residual_bytes": 96,
        "token_stream_bytes": INCUMBENT_TOKEN_BYTES,
    }
    for key, value in expected.items():
        if anatomy[key] != value:
            raise RuntimeError(f"DX2 anatomy drift for {key}: {anatomy[key]} != {value}")
    return anatomy, tokens


def spatial_orders() -> dict[str, np.ndarray]:
    flat = np.arange(SITES, dtype=np.int64).reshape(H, W)

    group = np.empty((H, W), dtype=np.int16)
    yy, xx = np.indices((H, W))
    group[:] = (xx % 64) + 2 * (yy % 64)
    rc64_event = np.argsort(group.reshape(-1), kind="stable")

    block8 = (
        flat.reshape(H // 8, 8, W // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )

    x = xx.reshape(-1).astype(np.uint32)
    y = yy.reshape(-1).astype(np.uint32)
    morton_key = np.zeros(SITES, dtype=np.uint32)
    for bit in range(10):
        morton_key |= ((x >> bit) & 1) << (2 * bit)
        morton_key |= ((y >> bit) & 1) << (2 * bit + 1)
    morton = np.argsort(morton_key, kind="stable")

    serpentine_rows = [flat[row] if row % 2 == 0 else flat[row, ::-1] for row in range(H)]
    serpentine = np.concatenate(serpentine_rows)

    orders = {
        "raster": flat.reshape(-1),
        "rc64_event": rc64_event,
        "block8": block8,
        "morton": morton,
        "serpentine": serpentine,
    }
    target = np.arange(SITES, dtype=np.int64)
    for name, order in orders.items():
        if not np.array_equal(np.sort(order), target):
            raise RuntimeError(f"{name} is not a spatial permutation")
    return orders


def profile_tokens(tokens: np.ndarray, orders: dict[str, np.ndarray]) -> dict[str, Any]:
    counts = np.zeros(ALPHABET, dtype=np.int64)
    temporal_equal = 0
    horizontal_equal = 0
    vertical_equal = 0
    comparisons_temporal = 0
    comparisons_horizontal = 0
    comparisons_vertical = 0
    adjacency: dict[str, dict[str, int]] = {
        name: {"equal": 0, "comparisons": 0} for name in orders
    }
    previous_last: dict[str, int | None] = dict.fromkeys(orders)
    previous_frame: np.ndarray | None = None
    for frame_index in range(T):
        frame = np.asarray(tokens[frame_index])
        counts += np.bincount(frame.reshape(-1), minlength=ALPHABET)
        horizontal_equal += int(np.count_nonzero(frame[:, 1:] == frame[:, :-1]))
        vertical_equal += int(np.count_nonzero(frame[1:, :] == frame[:-1, :]))
        comparisons_horizontal += H * (W - 1)
        comparisons_vertical += (H - 1) * W
        if previous_frame is not None:
            temporal_equal += int(np.count_nonzero(frame == previous_frame))
            comparisons_temporal += SITES
        previous_frame = frame.copy()
        flat = frame.reshape(-1)
        for name, order in orders.items():
            arranged = flat[order]
            adjacency[name]["equal"] += int(np.count_nonzero(arranged[1:] == arranged[:-1]))
            adjacency[name]["comparisons"] += SITES - 1
            if previous_last[name] is not None:
                adjacency[name]["equal"] += int(arranged[0] == previous_last[name])
                adjacency[name]["comparisons"] += 1
            previous_last[name] = int(arranged[-1])

    def fraction(equal: int, comparisons: int) -> float:
        return equal / comparisons

    for value in adjacency.values():
        value["fraction"] = fraction(value["equal"], value["comparisons"])
    return {
        "shape": [T, H, W],
        "dtype": "uint8",
        "alphabet_observed": np.flatnonzero(counts).tolist(),
        "class_counts": counts.tolist(),
        "class_fractions": (counts / counts.sum()).tolist(),
        "temporal_equal": {
            "equal": temporal_equal,
            "comparisons": comparisons_temporal,
            "fraction": fraction(temporal_equal, comparisons_temporal),
        },
        "horizontal_equal": {
            "equal": horizontal_equal,
            "comparisons": comparisons_horizontal,
            "fraction": fraction(horizontal_equal, comparisons_horizontal),
        },
        "vertical_equal": {
            "equal": vertical_equal,
            "comparisons": comparisons_vertical,
            "fraction": fraction(vertical_equal, comparisons_vertical),
        },
        "frame_stream_adjacency_by_spatial_order": adjacency,
    }


def order_shape_only(tokens: np.ndarray, order: np.ndarray, temporal_inner: bool) -> bytes:
    frames = np.asarray(tokens).reshape(T, SITES)
    arranged = frames[:, order]
    if temporal_inner:
        arranged = arranged.T
    return np.ascontiguousarray(arranged).tobytes()


def invert_shape_only(payload: bytes, order: np.ndarray, temporal_inner: bool) -> np.ndarray:
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != TOKEN_COUNT:
        raise RuntimeError("shape-only candidate changed symbol count")
    arranged = values.reshape(SITES, T).T if temporal_inner else values.reshape(T, SITES)
    restored = np.empty((T, SITES), dtype=np.uint8)
    restored[:, order] = arranged
    return restored.reshape(T, H, W)


def class_sorted_packet(tokens: np.ndarray) -> tuple[bytes, bytes, dict[str, Any]]:
    flat = np.asarray(tokens).reshape(-1)
    counts = np.bincount(flat, minlength=ALPHABET).astype(np.uint64)
    default = int(np.argmax(counts))
    sorted_symbols = np.repeat(np.arange(ALPHABET, dtype=np.uint8), counts.astype(np.int64))
    header = CLASS_HEADER.pack(
        b"CLS1", T, H, W, ALPHABET, default, *(int(value) for value in counts)
    )
    bitmaps: list[bytes] = []
    for class_id in range(ALPHABET):
        if class_id == default:
            continue
        bitmaps.append(np.packbits(flat == class_id, bitorder="little").tobytes())
    packet = header + sorted_symbols.tobytes() + b"".join(bitmaps)
    meta = {
        "default_class": default,
        "class_counts": counts.astype(int).tolist(),
        "sorted_symbol_bytes": len(sorted_symbols),
        "position_bitmap_bytes": sum(map(len, bitmaps)),
        "header_bytes": len(header),
        "rule_118": (
            "content-derived class grouping is not free; the packet counts the sorted symbols, "
            "the selected default in its header, and four explicit position bitmaps"
        ),
    }
    return packet, sorted_symbols.tobytes(), meta


def invert_class_sorted_packet(payload: bytes) -> np.ndarray:
    if len(payload) < CLASS_HEADER.size:
        raise RuntimeError("truncated CLS1 packet")
    magic, t, h, w, alphabet, default, *counts = CLASS_HEADER.unpack_from(payload)
    if (magic, t, h, w, alphabet) != (b"CLS1", T, H, W, ALPHABET):
        raise RuntimeError("CLS1 header drift")
    if default >= ALPHABET or sum(counts) != TOKEN_COUNT:
        raise RuntimeError("invalid CLS1 counts")
    cursor = CLASS_HEADER.size
    sorted_symbols = np.frombuffer(payload, dtype=np.uint8, count=TOKEN_COUNT, offset=cursor)
    cursor += TOKEN_COUNT
    expected_sorted = np.repeat(np.arange(ALPHABET, dtype=np.uint8), np.asarray(counts, dtype=np.int64))
    if not np.array_equal(sorted_symbols, expected_sorted):
        raise RuntimeError("CLS1 sorted stream is not canonical")
    bitmap_bytes = (TOKEN_COUNT + 7) // 8
    restored = np.full(TOKEN_COUNT, default, dtype=np.uint8)
    occupied = np.zeros(TOKEN_COUNT, dtype=bool)
    for class_id in range(ALPHABET):
        if class_id == default:
            continue
        bitmap = payload[cursor : cursor + bitmap_bytes]
        if len(bitmap) != bitmap_bytes:
            raise RuntimeError("truncated CLS1 bitmap")
        cursor += bitmap_bytes
        mask = np.unpackbits(
            np.frombuffer(bitmap, dtype=np.uint8), count=TOKEN_COUNT, bitorder="little"
        ).astype(bool)
        if np.any(occupied & mask):
            raise RuntimeError("overlapping CLS1 position bitmaps")
        occupied |= mask
        restored[mask] = class_id
    if cursor != len(payload):
        raise RuntimeError("CLS1 trailing bytes")
    if int(np.count_nonzero(~occupied)) != counts[default]:
        raise RuntimeError("CLS1 default count mismatch")
    return restored.reshape(T, H, W)


def compress_payload(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return bytes(brotli.compress(raw, quality=11))
    if coder == "lzma1_1m":
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    raise ValueError(coder)


def decompress_payload(coded: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return bytes(brotli.decompress(coded))
    if coder == "lzma1_1m":
        return lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    if coder == "zlib9":
        return zlib.decompress(coded)
    raise ValueError(coder)


def run_coder(raw: bytes, coder: str) -> tuple[str, bytes, bytes, float]:
    started = time.monotonic()
    coded = compress_payload(raw, coder)
    repeat = compress_payload(raw, coder)
    if coded != repeat:
        raise RuntimeError(f"non-deterministic coder: {coder}")
    if decompress_payload(coded, coder) != raw:
        raise RuntimeError(f"coder roundtrip failed: {coder}")
    return coder, coded, repeat, time.monotonic() - started


def validate_candidate_checkpoint(checkpoint: dict[str, Any]) -> None:
    for key in ("raw", "inverse"):
        validate_receipt(checkpoint[key])
    for result in checkpoint["coders"]:
        validate_receipt(result["payload"])
        validate_receipt(result["repeat"])


def deterministic_contract(value: Any) -> Any:
    """Remove runtime telemetry while preserving every semantic/payload field."""
    if isinstance(value, dict):
        return {
            key: deterministic_contract(item)
            for key, item in value.items()
            if not key.startswith("elapsed_seconds")
        }
    if isinstance(value, list):
        return [deterministic_contract(item) for item in value]
    return value


def retain_candidate(
    output: Path,
    name: str,
    raw: bytes,
    inverse: np.ndarray,
    source_sha: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    candidate_dir = output / "retained/candidates" / name
    checkpoint_path = candidate_dir / "candidate.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        validate_candidate_checkpoint(checkpoint)
        if (
            checkpoint["raw"]["bytes"] != len(raw)
            or checkpoint["raw"]["sha256"] != sha256_bytes(raw)
        ):
            raise RuntimeError(f"resumed raw candidate drift for {name}")
        if checkpoint["inverse"]["sha256"] != source_sha:
            raise RuntimeError(f"resumed inverse proof drift for {name}")
        if (
            checkpoint["name"] != name
            or deterministic_contract(checkpoint["metadata"])
            != deterministic_contract(metadata)
        ):
            raise RuntimeError(f"resumed candidate contract drift for {name}")
        return checkpoint

    raw_receipt = atomic_bytes(candidate_dir / "raw.bin", raw)
    inverse_payload = np.ascontiguousarray(inverse).tobytes()
    inverse_receipt = atomic_bytes(candidate_dir / "inverse_t600_h384_w512.u8", inverse_payload)
    if inverse_receipt["sha256"] != source_sha or len(inverse_payload) != TOKEN_COUNT:
        raise RuntimeError(f"exact inversion failed for {name}")

    coder_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(CODERS)) as executor:
        futures = {executor.submit(run_coder, raw, coder): coder for coder in CODERS}
        for future in as_completed(futures):
            coder, coded, repeat, elapsed = future.result()
            payload_receipt = atomic_bytes(candidate_dir / f"{coder}.bin", coded)
            repeat_receipt = atomic_bytes(candidate_dir / f"{coder}.repeat.bin", repeat)
            coder_results.append(
                {
                    "coder": coder,
                    "bytes": len(coded),
                    "elapsed_seconds_primary_plus_repeat": elapsed,
                    "payload": payload_receipt,
                    "repeat": repeat_receipt,
                }
            )
    coder_results.sort(key=lambda row: row["coder"])
    winner = min(coder_results, key=lambda row: (row["bytes"], row["coder"]))
    checkpoint = {
        "schema": "ddm.to2.candidate.v1",
        "name": name,
        "metadata": metadata,
        "raw": raw_receipt,
        "inverse": inverse_receipt,
        "inversion_exact": True,
        "coders": coder_results,
        "winner": {"coder": winner["coder"], "bytes": winner["bytes"]},
    }
    atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def prediction(profile: dict[str, Any]) -> dict[str, Any]:
    threshold = INCUMBENT_TOKEN_BYTES - int(np.ceil(0.10 * INCUMBENT_TOKEN_BYTES))
    return {
        "schema": "ddm.to2.prior_prediction.v1",
        "written_before_any_candidate_compression": True,
        "prediction": (
            "site_time with Brotli q11 will be the best tested generic shape-only ordering and will "
            "cut at least 10 percent from the 113777-byte shipped token member"
        ),
        "predicted_winner": "site_time",
        "predicted_coder": "brotli_q11",
        "success_threshold_bytes_inclusive": threshold,
        "mechanism": (
            "time is made the inner run at each fixed site, exposing repeated labels across adjacent "
            "pairs to a generic dictionary coder"
        ),
        "structural_basis": {
            "temporal_equal_fraction": profile["temporal_equal"]["fraction"],
            "horizontal_equal_fraction": profile["horizontal_equal"]["fraction"],
            "vertical_equal_fraction": profile["vertical_equal"]["fraction"],
        },
        "falsifier": (
            "no generic shape-only tested order is at or below the success threshold; the stronger "
            "charter falsifier is every generic order within about 2 percent of the incumbent"
        ),
        "newly_discovered_caveat_before_race": (
            "the shipped baseline is already a learned 19-member HPAC/RC64 coder, not generic Brotli, "
            "so the prior is a direct and deliberately difficult representation challenge"
        ),
    }


def score_projection(candidate_stream_bytes: int) -> dict[str, Any]:
    projected_archive = INCUMBENT_ARCHIVE_BYTES - INCUMBENT_TOKEN_BYTES + candidate_stream_bytes
    projected_score = FIXED_DISTORTION + RATE_WEIGHT * projected_archive / RATE_DENOMINATOR
    return {
        "projection_not_measurement": True,
        "optimistic_fixed_receiver_specialization_no_extra_selector_bytes": True,
        "projected_archive_bytes": projected_archive,
        "projected_score_at_fixed_distortion": projected_score,
        "delta_archive_bytes_vs_dx2": projected_archive - INCUMBENT_ARCHIVE_BYTES,
        "delta_score_vs_dx2": projected_score - INCUMBENT_SCORE,
        "under_137986_byte_ceiling": projected_archive <= ARCHIVE_CEILING,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed_parent = DEFAULT_OUTPUT.parent.resolve()
    if output.parent != allowed_parent:
        raise RuntimeError(f"output must be a direct child of {allowed_parent}")

    free_bytes = shutil.disk_usage(allowed_parent.parent).free
    required_free_bytes = 8 << 30
    if free_bytes < required_free_bytes:
        raise RuntimeError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")
    output.mkdir(parents=True, exist_ok=True)
    cleanup_entries = cleanup_incomplete_temps(output)

    pins = {
        str(path): verify_pin(path, expected_bytes, expected_sha)
        for path, (expected_bytes, expected_sha) in EXPECTED.items()
    }
    source_receipt = atomic_copy(DX2_TOKENS, output / "retained/input/dx2_tokens_decoded.u8")
    archive_receipt = atomic_copy(DX2_ARCHIVE, output / "retained/input/archive.zip")
    ad2_receipt = atomic_copy(AD2_RESULT, output / "retained/input/ad2_result.json")
    rb1_receipt = atomic_copy(RB1_MEMO, output / "retained/input/rb1_memo.md")
    producer_source = Path(__file__).resolve()
    producer_sha = sha256_file(producer_source)
    producer_receipt = atomic_copy(
        producer_source,
        output / "retained/source" / f"producer_{producer_sha[:16]}.py",
    )

    with zipfile.ZipFile(DX2_ARCHIVE) as archive:
        if archive.namelist() != ["p"] or archive.getinfo("p").compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("DX2 ZIP member contract drift")
        member = archive.read("p")
    anatomy, token_stream = parse_anatomy(member, DX2_ARCHIVE.stat().st_size)
    member_receipt = atomic_bytes(output / "retained/input/dx2_member.bin", member)
    stream_receipt = atomic_bytes(output / "retained/input/dx2_token_stream_rc64.bin", token_stream)
    if stream_receipt["bytes"] != INCUMBENT_TOKEN_BYTES:
        raise RuntimeError("incumbent token stream size drift")
    if stream_receipt["sha256"] != SHIPPED_STREAM_DIGEST:
        raise RuntimeError("incumbent token stream sha drift")

    decoder_checkpoint = json.loads(DX2_TOKEN_RECEIPT.read_text())
    binding = decoder_checkpoint["binding"]
    decoded = decoder_checkpoint["token_decoder"]
    tokens_receipt = decoder_checkpoint["tokens"]
    renderer_constants = binding["token_decoder_fingerprint"]["renderer_contract"]["constants"]
    if not decoder_checkpoint["complete"] or decoder_checkpoint["token_decoder"]["token_codec"] != "rc64":
        raise RuntimeError("decoded token checkpoint is not complete RC64 custody")
    if binding["archive_sha256"] != archive_receipt["sha256"]:
        raise RuntimeError("decoded token checkpoint archive binding drift")
    if binding["token_stream_sha256"] != stream_receipt["sha256"]:
        raise RuntimeError("decoded token checkpoint stream binding drift")
    if decoded["decoded_token_sha256"] != source_receipt["sha256"]:
        raise RuntimeError("decoded token checkpoint output binding drift")
    if tokens_receipt["bytes"] != TOKEN_COUNT or tokens_receipt["sha256"] != source_receipt["sha256"]:
        raise RuntimeError("decoded token checkpoint token receipt drift")
    if (
        binding["pair_count"],
        renderer_constants["N"],
        renderer_constants["EVAL_H"],
        renderer_constants["EVAL_W"],
        renderer_constants["NUM_CLASSES"],
    ) != (T, T, H, W, ALPHABET):
        raise RuntimeError("decoded token checkpoint shape/alphabet binding drift")
    decoder_receipt = atomic_copy(
        DX2_TOKEN_RECEIPT,
        output / "retained/input/tokens_cpu_stage_complete.json",
    )

    tokens = np.memmap(source_receipt["path"], dtype=np.uint8, mode="r", shape=(T, H, W))
    observed = np.unique(tokens)
    if not np.array_equal(observed, np.arange(ALPHABET, dtype=np.uint8)):
        raise RuntimeError(f"token alphabet drift: {observed}")
    orders = spatial_orders()

    profile_path = output / "PROFILE.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if (
            profile.get("schema") != "ddm.to2.profile.v1"
            or profile.get("source", {}).get("sha256") != source_receipt["sha256"]
            or profile.get("shape") != [T, H, W]
        ):
            raise RuntimeError("resumed profile contract drift")
    else:
        profile = profile_tokens(tokens, orders)
        profile.update(
            {
                "schema": "ddm.to2.profile.v1",
                "source": source_receipt,
                "actual_incumbent_traversal": (
                    "frame outer, then group 0..189 where group=(x mod 64)+2*(y mod 64), "
                    "then raster positions within each group"
                ),
                "actual_incumbent_coder": "19-member HPAC probability law plus RC64 arithmetic stream",
            }
        )
        atomic_json(profile_path, profile)

    prediction_path = output / "PREDICTION.json"
    if prediction_path.exists():
        prior = json.loads(prediction_path.read_text())
        if (
            prior.get("schema") != "ddm.to2.prior_prediction.v1"
            or not prior.get("written_before_any_candidate_compression")
            or prior.get("structural_basis", {}).get("temporal_equal_fraction")
            != profile["temporal_equal"]["fraction"]
        ):
            raise RuntimeError("resumed prior-prediction contract drift")
    else:
        prior = prediction(profile)
        atomic_json(prediction_path, prior)

    state_path = output / "STATE.json"
    candidates: list[dict[str, Any]] = []
    specs: list[tuple[str, str, bool]] = [
        ("frame_raster", "raster", False),
        ("rc64_event_order_generic_coders", "rc64_event", False),
        ("site_time", "raster", True),
        ("block8_frame", "block8", False),
        ("block8_time", "block8", True),
        ("morton_frame", "morton", False),
        ("morton_time", "morton", True),
        ("serpentine_time", "serpentine", True),
    ]
    for index, (name, order_name, temporal_inner) in enumerate(specs):
        raw = order_shape_only(tokens, orders[order_name], temporal_inner)
        inverse = invert_shape_only(raw, orders[order_name], temporal_inner)
        candidate = retain_candidate(
            output,
            name,
            raw,
            inverse,
            source_receipt["sha256"],
            {
                "family": "generic_shape_only_permutation",
                "spatial_order": order_name,
                "temporal_inner": temporal_inner,
                "rule_118": (
                    "permutation is fixed by public shape and generic algorithm; no content-derived "
                    "table or learned data is treated as free"
                ),
            },
        )
        candidates.append(candidate)
        atomic_json(
            state_path,
            {
                "schema": "ddm.to2.state.v1",
                "stage": "race",
                "completed_candidates": [row["name"] for row in candidates],
                "remaining_candidates": [row[0] for row in specs[index + 1 :]]
                + ["class_sorted_counted_positions"],
            },
            replace=True,
        )

    class_raw, gross_sorted, class_meta = class_sorted_packet(tokens)
    gross_dir = output / "retained/diagnostics/class_sorted_gross_only"
    gross_receipt = atomic_bytes(gross_dir / "raw.bin", gross_sorted)
    gross_coders: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(CODERS)) as executor:
        futures = {executor.submit(run_coder, gross_sorted, coder): coder for coder in CODERS}
        for future in as_completed(futures):
            coder, coded, repeat, elapsed = future.result()
            gross_coders.append(
                {
                    "coder": coder,
                    "bytes": len(coded),
                    "elapsed_seconds_primary_plus_repeat": elapsed,
                    "payload": atomic_bytes(gross_dir / f"{coder}.bin", coded),
                    "repeat": atomic_bytes(gross_dir / f"{coder}.repeat.bin", repeat),
                }
            )
    gross_coders.sort(key=lambda row: row["coder"])
    inverse = invert_class_sorted_packet(class_raw)
    class_candidate = retain_candidate(
        output,
        "class_sorted_counted_positions",
        class_raw,
        inverse,
        source_receipt["sha256"],
        {
            "family": "content_derived_class_sort_with_counted_inverse_map",
            **class_meta,
            "gross_only_diagnostic": {
                "admissible": False,
                "reason": "the sorted symbols alone cannot reconstruct positions",
                "raw": gross_receipt,
                "coders": gross_coders,
            },
        },
    )
    candidates.append(class_candidate)

    generic_candidates = [
        candidate
        for candidate in candidates
        if candidate["metadata"]["family"] == "generic_shape_only_permutation"
    ]
    generic_winner = min(
        generic_candidates,
        key=lambda row: (row["winner"]["bytes"], row["name"], row["winner"]["coder"]),
    )
    overall_winner = min(
        candidates,
        key=lambda row: (row["winner"]["bytes"], row["name"], row["winner"]["coder"]),
    )
    generic_threshold = prior["success_threshold_bytes_inclusive"]
    every_generic_within_two_percent = all(
        abs(candidate["winner"]["bytes"] - INCUMBENT_TOKEN_BYTES) / INCUMBENT_TOKEN_BYTES <= 0.02
        for candidate in generic_candidates
    )
    comparison_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        winner_bytes = candidate["winner"]["bytes"]
        comparison_rows.append(
            {
                "candidate": candidate["name"],
                "raw_bytes": candidate["raw"]["bytes"],
                "winner_coder": candidate["winner"]["coder"],
                "winner_bytes": winner_bytes,
                "delta_bytes_vs_shipped_rc64": winner_bytes - INCUMBENT_TOKEN_BYTES,
                "fractional_delta_vs_shipped_rc64": winner_bytes / INCUMBENT_TOKEN_BYTES - 1,
                "projection": score_projection(winner_bytes),
                "inversion_exact": candidate["inversion_exact"],
                "rule_118_family": candidate["metadata"]["family"],
            }
        )
    result = {
        "schema": "ddm.to2.result.v1",
        "axis": "[macOS-CPU advisory] scorer-free exact token representation measurement",
        "authority": {
            "score_authority": False,
            "scorers_loaded": False,
            "archive_built": False,
            "all_archive_and_score_values_for_candidates_are_projections": True,
        },
        "command": {"argv": sys.argv, "cwd": str(Path.cwd())},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "brotli": getattr(brotli, "__version__", "unknown"),
        },
        "storage_preflight": {
            "tier": "/Volumes/VertigoDataTier/pact",
            "free_bytes_before": free_bytes,
            "required_free_bytes": required_free_bytes,
            "passed": True,
            "certified_incomplete_temp_cleanup": cleanup_entries,
            "auto_cleanup": (
                "no scratch payload is created outside the retained tree; immutable candidate stages "
                "are resumable and differing existing bytes fail closed"
            ),
        },
        "pins": pins,
        "retained_inputs": {
            "tokens": source_receipt,
            "token_decoder_checkpoint": decoder_receipt,
            "archive": archive_receipt,
            "archive_member": member_receipt,
            "shipped_token_stream": stream_receipt,
            "ad2_result": ad2_receipt,
            "rb1_memo": rb1_receipt,
            "producer": producer_receipt,
        },
        "anatomy": anatomy,
        "token_contract": {
            "shape": [T, H, W],
            "count": TOKEN_COUNT,
            "alphabet": observed.astype(int).tolist(),
            "source_sha256": source_receipt["sha256"],
            "every_candidate_inverse_sha256": source_receipt["sha256"],
        },
        "profile": profile,
        "prior_prediction": prior,
        "incumbent": {
            "representation": "shipped 19-member HPAC plus RC64 arithmetic token stream",
            "token_bytes": INCUMBENT_TOKEN_BYTES,
            "archive_bytes": INCUMBENT_ARCHIVE_BYTES,
            "archive_sha256": EXPECTED[DX2_ARCHIVE][1],
            "exact_score": INCUMBENT_SCORE,
            "score_axis": "[contest-CUDA T4 n600]",
        },
        "candidate_rows": comparison_rows,
        "generic_winner": {
            "candidate": generic_winner["name"],
            **generic_winner["winner"],
            "beats_shipped_rc64": generic_winner["winner"]["bytes"] < INCUMBENT_TOKEN_BYTES,
            "meets_prior_10_percent_cut": generic_winner["winner"]["bytes"] <= generic_threshold,
        },
        "overall_tested_winner": {
            "candidate": overall_winner["name"],
            **overall_winner["winner"],
            "beats_shipped_rc64": overall_winner["winner"]["bytes"] < INCUMBENT_TOKEN_BYTES,
        },
        "falsifier": {
            "prior_10_percent_cut_falsified": generic_winner["winner"]["bytes"] > generic_threshold,
            "stronger_every_generic_within_about_2_percent": every_generic_within_two_percent,
            "two_percent_denominator": INCUMBENT_TOKEN_BYTES,
        },
        "fixed_distortion_projection_constants": {
            "fixed_distortion": FIXED_DISTORTION,
            "archive_ceiling_bytes": ARCHIVE_CEILING,
            "bytes_needed_from_dx2": INCUMBENT_ARCHIVE_BYTES - ARCHIVE_CEILING,
            "score_per_byte": RATE_WEIGHT / RATE_DENOMINATOR,
        },
        "candidates": candidates,
        "outcome_scope": (
            "exact DX2 token field on the pinned archive and the tested generic shape-only "
            "permutations with Brotli q11, raw LZMA1 1 MiB, and zlib9"
        ),
    }
    result_receipt = atomic_json(output / "RESULT.json", result, replace=True)
    manifest = {
        "schema": "ddm.to2.manifest.v1",
        "result": result_receipt,
        "payloads": sorted(
            [file_receipt(path) for path in (output / "retained").rglob("*") if path.is_file()],
            key=lambda row: row["path"],
        ),
    }
    manifest_receipt = atomic_json(output / "MANIFEST.json", manifest, replace=True)
    atomic_json(
        state_path,
        {
            "schema": "ddm.to2.state.v1",
            "stage": "complete",
            "completed_candidates": [row["name"] for row in candidates],
            "result": result_receipt,
            "manifest": manifest_receipt,
        },
        replace=True,
    )
    print(json.dumps({"result": result_receipt, "manifest": manifest_receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
