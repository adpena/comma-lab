#!/usr/bin/env python3
"""Build the scorer-free T1 whole-container rehearsal on retained pass objects.

The adapter re-encodes the retained C1 semantic plane under CP135's exact HP3
probability object, converts one selected CPR1 pose carrier losslessly into the
F26 CAP1 wire, rebuilds the complete split-Brotli/RC64 container, and proves
receiver parse-back.  Every materialized payload is retained below ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_ps135_pose_resolve as ps135

DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_t1r1")
CP135_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip")
CP135_RUNTIME = CP135_ARCHIVE.parent
PR135_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip")
C1_SPATIAL = Path("/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/c1_solved_tokens_n600.u8")
PASS4_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/leg_a/passes/pass_04/selected")
PASS4_ARCHIVE = PASS4_ROOT / "archive.zip"
PASS4_CARRIER = PASS4_ROOT / "carrier.cpr1"
PASS4_COEFFICIENTS = PASS4_ROOT / "coefficients.int16.npy"
EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")
BROTLI = shutil.which("brotli") or "brotli"

CP135_BYTES = 186_252
CP135_SHA256 = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_RUNTIME_TREE_SHA256 = "f56139f82447018765879ce5ba6138d087911dacaa1f181a79cabea03b790996"
PR135_BYTES = 186_724
PR135_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
C1_BYTES = 117_964_800
C1_SHA256 = "2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5"
PASS4_ARCHIVE_BYTES = 187_223
PASS4_ARCHIVE_SHA256 = "e269d1ffbe0bf56ec8471a6869b7ec081f3de07e852b193aa251a963c543becb"
PASS4_CARRIER_BYTES = 23_051
PASS4_CARRIER_SHA256 = "4c1a65c7f3a9bfa1b0f7677494ddbfdad87881fe0f4b78613893bd555f725ef2"
PASS4_COEFFICIENT_BYTES = 14_528
PASS4_COEFFICIENT_SHA256 = "da9bba74fdaadc8110b9eb0614decb6d3a5caa076a03b01eee5647d32c37590e"
EVENTS_PER_FRAME = 384 * 512
FRAMES = 600
PACKED_CAP1_FLAG = 1 << 15
MIN_FREE_BYTES = 2 * 1024**3
AXIS = "[macOS-CPU scorer-free byte/custody apparatus]"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError(f"{label} differs from its rehearsal pin: {path}")


def atomic_bytes(path: Path, value: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def raw_array_sha256(array: np.ndarray, *, rows: int = FRAMES) -> str:
    digest = hashlib.sha256()
    for index in range(rows):
        digest.update(np.asarray(array[index]).tobytes())
    return digest.hexdigest()


def retained_tree_record(root: Path) -> dict[str, Any]:
    """Hash durable evidence while excluding this manifest and filesystem metadata."""

    excluded = {"99_TREE_MANIFEST.json"}
    rows = []
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if (
            relative in excluded
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path.name.startswith("._")
        ):
            continue
        record = file_record(path)
        rows.append({"relative_path": relative, **record})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\n")
    return {
        "schema": "ddm_t1r1_retained_tree.v1",
        "root": str(root.resolve()),
        "files": rows,
        "file_count": len(rows),
        "tree_sha256": digest.hexdigest(),
        "excluded_relative_paths": sorted(excluded),
        "excluded_patterns": ["**/__pycache__/**", "**/*.pyc", "**/._*"],
        "exclusion_reason": "manifest self-reference and non-payload filesystem metadata",
    }


def retained_root(output: Path) -> Path:
    return output / "retained"


def hp3_adapter_root(output: Path, event_sha256: str) -> Path:
    if len(event_sha256) != 64 or any(value not in "0123456789abcdef" for value in event_sha256):
        raise RuntimeError("HP3 adapter source digest is not a lowercase SHA-256")
    return retained_root(output) / "hp3_reencode_adapter" / event_sha256


def validate_sources() -> dict[str, Any]:
    require_file(CP135_ARCHIVE, size=CP135_BYTES, digest=CP135_SHA256, label="CP135 archive")
    require_file(PR135_ARCHIVE, size=PR135_BYTES, digest=PR135_SHA256, label="PR135 archive")
    require_file(C1_SPATIAL, size=C1_BYTES, digest=C1_SHA256, label="C1 solved plane")
    require_file(
        PASS4_ARCHIVE,
        size=PASS4_ARCHIVE_BYTES,
        digest=PASS4_ARCHIVE_SHA256,
        label="pass-4 selected archive",
    )
    require_file(
        PASS4_CARRIER,
        size=PASS4_CARRIER_BYTES,
        digest=PASS4_CARRIER_SHA256,
        label="pass-4 CPR1 carrier",
    )
    require_file(
        PASS4_COEFFICIENTS,
        size=PASS4_COEFFICIENT_BYTES,
        digest=PASS4_COEFFICIENT_SHA256,
        label="pass-4 coefficients",
    )
    if not CP135_RUNTIME.is_dir() or not EXPERIMENT_BOOK.is_dir():
        raise RuntimeError("CP135 runtime or PR135 ExperimentBook is unavailable")
    runtime_tree = cp135.tree_record(CP135_RUNTIME)
    if runtime_tree["tree_sha256"] != CP135_RUNTIME_TREE_SHA256:
        raise RuntimeError("CP135 runtime tree differs from its promoted parse-back receipt")
    return runtime_tree


def cx2_tm1_parseback() -> dict[str, Any]:
    source = ps135.load_lc2_source()
    carrier = PASS4_CARRIER.read_bytes()
    result = ps135.parse_candidate_archive(PASS4_ARCHIVE.read_bytes(), carrier, source)
    state = ps135.decode_carrier(carrier)
    coefficients = np.load(PASS4_COEFFICIENTS, allow_pickle=False)
    if coefficients.dtype != np.int16 or coefficients.shape != (FRAMES, 12):
        raise RuntimeError("pass-4 coefficient payload has the wrong dtype or shape")
    if not np.array_equal(coefficients, state.codes):
        raise RuntimeError("pass-4 direct coefficients differ from CX2/TM1 carrier parse-back")
    result["direct_carrier"] = file_record(PASS4_CARRIER)
    result["direct_coefficients"] = file_record(PASS4_COEFFICIENTS)
    result["direct_coefficients_equal_parsed_codes"] = True
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_runtime_tree = validate_sources()
    retained = retained_root(args.output)
    retained.mkdir(parents=True, exist_ok=True)
    storage_mount = Path("/Volumes/APDataStore")
    if not args.output.resolve().is_relative_to(storage_mount):
        raise RuntimeError("T1R1 output must remain on the granted APDataStore tier")
    usage = shutil.disk_usage(storage_mount)
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError("APDataStore storage preflight failed closed")
    storage = {
        "schema": "ddm_t1r1_storage_preflight.v1",
        "path": str(storage_mount),
        "free_bytes": usage.free,
        "required_free_bytes": MIN_FREE_BYTES,
        "status": "PASS",
    }
    atomic_json(retained / "00_STORAGE_PREFLIGHT.json", storage)
    atomic_json(retained / "05_CP135_RUNTIME_SOURCE_TREE.json", source_runtime_tree)

    source_parseback = cx2_tm1_parseback()
    atomic_json(retained / "10_CX2_TM1_SOURCE_PARSEBACK.json", source_parseback)

    chunks = retained / "c1_event_order"
    chunks.mkdir(parents=True, exist_ok=True)
    event_path = chunks / "c1_solved_tokens.f26_event_order.npy"
    progress_path = chunks / "PREPARE_PROGRESS.json"
    positions = np.concatenate(cp135._group_positions(CP135_RUNTIME))
    if positions.shape != (EVENTS_PER_FRAME,) or np.unique(positions).size != EVENTS_PER_FRAME:
        raise RuntimeError("F26 group positions are not a permutation of one token frame")
    positions_sha256 = sha256_bytes(positions.astype("<i8", copy=False).tobytes())
    source_binding = {
        "source_spatial_sha256": C1_SHA256,
        "group_positions_sha256": positions_sha256,
    }
    spatial = np.memmap(C1_SPATIAL, mode="r", dtype=np.uint8, shape=(FRAMES, 384, 512))
    start = 0
    if event_path.is_file() and not progress_path.is_file():
        event_order = np.lib.format.open_memmap(event_path, mode="r+")
        if event_order.dtype != np.uint8 or event_order.shape != (FRAMES * EVENTS_PER_FRAME,) or np.any(event_order):
            raise RuntimeError("orphan C1 event-order payload is not an adoptable empty initialization")
        progress = {
            "schema": "ddm_t1r1_c1_event_order_progress.v1",
            "next_frame": 0,
            "complete": False,
            "prefix_event_sha256": hashlib.sha256().hexdigest(),
            **source_binding,
        }
        atomic_json(progress_path, progress)
    elif progress_path.is_file() and not event_path.is_file():
        raise RuntimeError("C1 event-order progress receipt has no payload")
    if event_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if any(progress.get(key) != value for key, value in source_binding.items()):
            raise RuntimeError("C1 event-order checkpoint is bound to a different source")
        start = int(progress["next_frame"])
        if not 0 <= start <= FRAMES or bool(progress.get("complete")) != (start == FRAMES):
            raise RuntimeError("C1 event-order progress marker is invalid")
        event_order = np.lib.format.open_memmap(event_path, mode="r+")
        if event_order.dtype != np.uint8 or event_order.shape != (FRAMES * EVENTS_PER_FRAME,):
            raise RuntimeError("retained C1 event-order checkpoint has invalid geometry")
    else:
        event_order = np.lib.format.open_memmap(
            event_path,
            mode="w+",
            dtype=np.uint8,
            shape=(FRAMES * EVENTS_PER_FRAME,),
        )
        progress = {
            "schema": "ddm_t1r1_c1_event_order_progress.v1",
            "next_frame": 0,
            "complete": False,
            "prefix_event_sha256": hashlib.sha256().hexdigest(),
            **source_binding,
        }
        atomic_json(progress_path, progress)
    prefix_digest = hashlib.sha256()
    for frame in range(start):
        first = frame * EVENTS_PER_FRAME
        prefix_digest.update(np.asarray(event_order[first : first + EVENTS_PER_FRAME]).tobytes())
    if prefix_digest.hexdigest() != progress.get("prefix_event_sha256"):
        raise RuntimeError("C1 event-order checkpoint prefix failed custody")
    for frame in range(start, FRAMES):
        flat = np.asarray(spatial[frame]).reshape(-1)
        events = flat[positions]
        inverse = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
        inverse[positions] = events
        if not np.array_equal(inverse, flat):
            raise RuntimeError(f"C1 event-order inverse failed at frame {frame}")
        first = frame * EVENTS_PER_FRAME
        event_order[first : first + EVENTS_PER_FRAME] = events
        prefix_digest.update(events.tobytes())
        if (frame + 1) % 24 == 0 or frame == FRAMES - 1:
            event_order.flush()
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_t1r1_c1_event_order_progress.v1",
                    "next_frame": frame + 1,
                    "complete": frame + 1 == FRAMES,
                    "prefix_event_sha256": prefix_digest.hexdigest(),
                    **source_binding,
                },
            )
            print(json.dumps({"prepared_frames": frame + 1}), flush=True)
    event_order = np.load(event_path, mmap_mode="r", allow_pickle=False)
    event_digest = raw_array_sha256(event_order.reshape(FRAMES, EVENTS_PER_FRAME))
    if event_digest != prefix_digest.hexdigest():
        raise RuntimeError("C1 event-order terminal digest differs from its checkpoint chain")
    if raw_array_sha256(spatial) != C1_SHA256:
        raise RuntimeError("C1 spatial payload changed during event-order materialization")
    manifest = {
        "schema": "ddm_t1r1_c1_event_order_manifest.v1",
        "complete": True,
        "chunks": [
            {
                "start_frame": 0,
                "end_frame": FRAMES,
                "symbols_path": str(event_path.resolve()),
                "symbols_sha256": sha256_file(event_path),
                "symbols_bytes": event_path.stat().st_size,
                "tokens": FRAMES * EVENTS_PER_FRAME,
            }
        ],
    }
    manifest_path = chunks / "chunk_manifest.json"
    atomic_json(manifest_path, manifest)
    result = {
        "schema": "ddm_t1r1_prepare_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "sources": {
            "cp135_archive": file_record(CP135_ARCHIVE),
            "pr135_archive": file_record(PR135_ARCHIVE),
            "c1_spatial": file_record(C1_SPATIAL),
            "pass4_archive": file_record(PASS4_ARCHIVE),
            "pass4_carrier": file_record(PASS4_CARRIER),
            "pass4_coefficients": file_record(PASS4_COEFFICIENTS),
        },
        "cx2_tm1_parseback": file_record(retained / "10_CX2_TM1_SOURCE_PARSEBACK.json"),
        "cp135_runtime_source_tree": file_record(retained / "05_CP135_RUNTIME_SOURCE_TREE.json"),
        "event_order_payload": file_record(event_path),
        "event_order_raw_sha256": event_digest,
        "spatial_raw_sha256": C1_SHA256,
        "source_manifest": file_record(manifest_path),
        "inverse_permutation_frames": FRAMES,
    }
    atomic_json(retained / "20_PREPARE_RESULT.json", result)
    return result


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        started = time.time()
        log.write(json.dumps({"argv": command, "started_unix_s": started}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        log.write(
            json.dumps(
                {
                    "argv": command,
                    "returncode": returncode,
                    "wall_s": time.time() - started,
                }
            )
            + "\n"
        )
        log.flush()
        os.fsync(log.fileno())
        if returncode:
            raise RuntimeError(f"subprocess failed rc={returncode}: {command}")


def reencode(args: argparse.Namespace) -> dict[str, Any]:
    retained = retained_root(args.output)
    prepared = json.loads((retained / "20_PREPARE_RESULT.json").read_text())
    manifest = Path(prepared["source_manifest"]["path"])
    event_sha = prepared["event_order_raw_sha256"]
    logs = retained / "logs"
    adapter_root = hp3_adapter_root(args.output, event_sha)
    common = [
        sys.executable,
        str(REPO / "experiments/ddm_cp135_rate_compose.py"),
        "--variant",
        "hp3_step2",
        "--archive",
        str(PR135_ARCHIVE),
        "--runtime",
        str(CP135_RUNTIME),
        "--output",
        str(adapter_root),
        "--dt1-manifest",
        str(manifest),
        "--experiment-book",
        str(EXPERIMENT_BOOK),
        "--expected-event-order-sha256",
        event_sha,
        "--expected-spatial-token-sha256",
        C1_SHA256,
    ]
    export_result = adapter_root / "retained/probabilities/hp3_step2/EXPORT_RESULT.json"
    command = common.copy()
    command.insert(2, "export")
    run_logged(command, logs / "30_HP3_EXPORT.log")
    rc64_result = adapter_root / "retained/coders/hp3_step2/FRESH_RC64_RESULT.json"
    command = common.copy()
    command.insert(2, "encode-rc64")
    run_logged(command, logs / "40_HP3_RC64.log")
    export = json.loads(export_result.read_text())
    rc64 = json.loads(rc64_result.read_text())
    token_payload = Path(rc64["token_payload"]["path"])
    probability_identity = export.get("probability_identity")
    if (
        export.get("complete_n600") is not True
        or export.get("source_symbol_sha256") != event_sha
        or probability_identity is None
        or file_record(Path(probability_identity["path"])) != probability_identity
        or rc64.get("symbol_identity") is not True
        or rc64.get("decoded_event_order_sha256") != event_sha
        or rc64.get("decoded_spatial_token_sha256") != C1_SHA256
        or file_record(token_payload) != rc64.get("token_payload")
    ):
        raise RuntimeError("HP3 C1 re-encode did not close at n600")
    result = {
        "schema": "ddm_t1r1_hp3_reencode_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "probability_export": file_record(export_result),
        "token_result": file_record(rc64_result),
        "token_payload": rc64["token_payload"],
        "decoded_event_order_sha256": rc64["decoded_event_order_sha256"],
        "decoded_spatial_token_sha256": rc64["decoded_spatial_token_sha256"],
        "events": rc64["events"],
        "encode_wall_s": rc64["encode_wall_s"],
        "decode_wall_s": rc64["decode_wall_s"],
    }
    atomic_json(retained / "50_HP3_REENCODE_RESULT.json", result)
    return result


def pack_dynamic_cap1_metadata(carrier_selector: bytes) -> tuple[bytes, dict[str, Any]]:
    if len(carrier_selector) < 183:
        raise RuntimeError("dynamic CAP1 carrier/selector section is truncated")
    bit_counts = carrier_selector[:6]
    scales = carrier_selector[6:102]
    factors = np.frombuffer(carrier_selector[102:126], dtype="<i2").astype(np.int16)
    biases = np.frombuffer(carrier_selector[126:138], dtype=np.int8).astype(np.int16)
    lengths = np.frombuffer(carrier_selector[138:170], dtype=np.uint8).astype(np.int16)
    ks = np.frombuffer(carrier_selector[170:182], dtype=np.uint8).astype(np.int16)
    if (
        np.any(factors < 0)
        or int(factors.min()) > 255
        or np.any(factors > 512)
        or np.any(factors - int(factors.min()) > 127)
        or np.any(biases < -16)
        or np.any(biases > 16)
        or np.any(lengths > 15)
        or np.any(ks >= 12)
        or np.any(ks - int(ks.min()) > 1)
    ):
        raise RuntimeError("dynamic CAP1 metadata exceeds the CP135 packed domains")
    packed = (
        bit_counts
        + scales
        + bytes((int(factors.min()),))
        + cp135._pack_unsigned(factors - int(factors.min()), 7)
        + cp135._pack_unsigned(biases & 0x3F, 6)
        + cp135._pack_unsigned(lengths, 4)
        + bytes((int(ks.min()),))
        + cp135._pack_unsigned(ks - int(ks.min()), 1)
        + carrier_selector[182:]
    )
    restored = unpack_dynamic_cap1_metadata(packed)
    if restored != carrier_selector:
        raise RuntimeError("dynamic packed CAP1 inverse differs")
    return packed, {
        "source_bytes": len(carrier_selector),
        "source_sha256": sha256_bytes(carrier_selector),
        "packed_bytes": len(packed),
        "packed_sha256": sha256_bytes(packed),
        "raw_delta_bytes": len(packed) - len(carrier_selector),
    }


def unpack_dynamic_cap1_metadata(packed: bytes) -> bytes:
    if len(packed) < 143:
        raise RuntimeError("dynamic packed CAP1 section is truncated")
    factor_base = packed[102]
    factors = factor_base + cp135._unpack_unsigned(packed[103:114], 12, 7)
    bias_codes = cp135._unpack_unsigned(packed[114:123], 12, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = cp135._unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
    k_base = packed[139]
    ks = (k_base + cp135._unpack_unsigned(packed[140:142], 12, 1)).astype(np.uint8)
    return (
        packed[:102]
        + factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
        + packed[142:]
    )


def cap1_to_f24s_body(cap1: bytes) -> bytes:
    if len(cap1) < 191 or cap1[:8] != b"CAP1\x01\x00\x00\x00":
        raise RuntimeError("CAP1 source is malformed")
    return cap1[8:14] + cap1[50:146] + cap1[14:50] + cap1[146:]


def generalize_residual_archive_source(source: str) -> str:
    old_guard = (
        "    if len(packed) != PACKED_CAP1_SECTION_BYTES:\n"
        '        raise ResidualArchiveError("packed CAP1 section has the wrong length")\n'
    )
    new_guard = '    if len(packed) < 143:\n        raise ResidualArchiveError("packed CAP1 field is truncated")\n'
    old_result = (
        "    if len(result) != CANONICAL_CAP1_SECTION_BYTES:\n"
        '        raise ResidualArchiveError("packed CAP1 inverse produced the wrong length")\n'
    )
    new_result = (
        "    if len(result) != len(packed) + 40:\n"
        '        raise ResidualArchiveError("packed CAP1 inverse produced the wrong length")\n'
    )
    if source.count(old_guard) != 1 or source.count(old_result) != 1:
        raise RuntimeError("CP135 receiver metadata guard drifted")
    source = source.replace(old_guard, new_guard).replace(old_result, new_result)
    start = source.index("def _decode_split_models(")
    end = source.index("\n\n@dataclass", start)
    replacement = '''def _decode_split_models(outer: bytes) -> tuple[bytes, bytes] | None:
    """Decode CP135 or the explicitly tagged T1R1 dynamic CAP1 split form."""
    if len(outer) < SPLIT_MODEL_HEADER.size:
        return None
    encoded_lengths = SPLIT_MODEL_HEADER.unpack_from(outer)
    packed_cap1 = bool(encoded_lengths[2] & PACKED_CAP1_LENGTH_FLAG)
    lengths = (encoded_lengths[0], encoded_lengths[1], encoded_lengths[2] & ~PACKED_CAP1_LENGTH_FLAG)
    model_end = SPLIT_MODEL_HEADER.size + sum(lengths)
    if min(lengths) <= 0 or model_end + 96 >= len(outer):
        return None
    offset = SPLIT_MODEL_HEADER.size
    streams = []
    for length in lengths:
        streams.append(outer[offset : offset + length])
        offset += length
    try:
        sections = tuple(_decompress_brotli(stream) for stream in streams)
    except ResidualArchiveError:
        return None
    if tuple(map(len, sections[:2])) != SPLIT_MODEL_SECTION_BYTES:
        return None
    carrier = sections[2]
    if packed_cap1:
        carrier = _restore_packed_cap1_metadata(carrier)
    elif len(carrier) == PACKED_CAP1_SECTION_BYTES:
        carrier = _restore_packed_cap1_metadata(carrier)
    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:
        return None
    return b"F24S" + sections[0] + sections[1] + carrier, outer[model_end:]
'''
    source = source[:start] + replacement + source[end:]
    marker = "CANONICAL_CAP1_SECTION_BYTES = 22_223\n"
    if source.count(marker) != 1:
        raise RuntimeError("CP135 receiver constant surface drifted")
    source = source.replace(marker, marker + "PACKED_CAP1_LENGTH_FLAG = 1 << 15\n")
    compile(source, "t1r1_residual_archive.py", "exec")
    return source


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def copy_runtime(destination: Path, archive: bytes) -> dict[str, Any]:
    for source_path in sorted(CP135_RUNTIME.rglob("*")):
        if not source_path.is_file() or "__pycache__" in source_path.parts or source_path.suffix == ".pyc":
            continue
        relative = source_path.relative_to(CP135_RUNTIME)
        if relative.as_posix() in {"archive.zip", "inflate.py"}:
            continue
        target = destination / relative
        value = source_path.read_bytes()
        if target.is_file() and target.read_bytes() != value and relative.as_posix() != "runtime/residual_archive.py":
            raise RuntimeError(f"retained runtime file changed: {target}")
        atomic_bytes(target, value, executable=os.access(source_path, os.X_OK))
    residual_source = (CP135_RUNTIME / "runtime/residual_archive.py").read_text()
    atomic_bytes(
        destination / "runtime/residual_archive.py",
        generalize_residual_archive_source(residual_source).encode(),
    )
    archive_path = destination / "archive.zip"
    atomic_bytes(archive_path, archive)
    inflate_source = (CP135_RUNTIME / "inflate.py").read_text()
    if inflate_source.count(CP135_SHA256) != 1 or inflate_source.count(f"ARCHIVE_BYTES = {CP135_BYTES:_}") != 1:
        raise RuntimeError("CP135 inflate.py pin surface drifted")
    inflate_source = inflate_source.replace(CP135_SHA256, sha256_bytes(archive)).replace(
        f"ARCHIVE_BYTES = {CP135_BYTES:_}", f"ARCHIVE_BYTES = {len(archive):_}"
    )
    if CP135_SHA256 in inflate_source or f"ARCHIVE_BYTES = {CP135_BYTES:_}" in inflate_source:
        raise RuntimeError("inflate.py pin adaptation failed")
    atomic_bytes(destination / "inflate.py", inflate_source.encode(), executable=True)
    return {
        "source_runtime": str(CP135_RUNTIME),
        "archive": file_record(archive_path),
        "residual_archive_adapter": file_record(destination / "runtime/residual_archive.py"),
        "inflate_entrypoint": file_record(destination / "inflate.py"),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    retained = retained_root(args.output)
    reencode_receipt = json.loads((retained / "50_HP3_REENCODE_RESULT.json").read_text())
    token_path = Path(reencode_receipt["token_payload"]["path"])
    token = token_path.read_bytes()
    cp_runtime = cp135.load_runtime(CP135_RUNTIME)
    pr_parts = cp_runtime.read_residual_archive(PR135_ARCHIVE)
    cp_parts = cp_runtime.read_residual_archive(CP135_ARCHIVE)
    base_models = cp135._base_physical_models(PR135_ARCHIVE)
    hp3, hp3_report = cp135.step2_ihs2(pr_parts.hpac_blob)
    _, hpac_body, semantic_body, _ = cp135._physical_model_parts(pr_parts, hp3, base_models)

    sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
    try:
        cap_module = importlib.import_module("cpr1_sub4.entropy.coefficient_ar1_codec")
    finally:
        sys.path.pop(0)
    pass4_carrier = PASS4_CARRIER.read_bytes()
    cap1, cap1_report = cap_module.encode_cap1(pass4_carrier, frames=FRAMES, dimensions=12)
    if cap_module.decode_cap1(cap1, frames=FRAMES, dimensions=12) != pass4_carrier:
        raise RuntimeError("pass-4 CPR1 to CAP1 conversion failed exact restoration")
    carrier_module = importlib.import_module("runtime.carrier_repack")
    _, selector = carrier_module.split_frame0_selector_carrier(cp_parts.carrier_blob)
    if selector is None or not selector.startswith(b"F0E1\x01"):
        raise RuntimeError("CP135 sparse frame-0 selector is unavailable")
    carrier_selector = cap1_to_f24s_body(cap1) + selector[5:]
    packed_carrier, pack_report = pack_dynamic_cap1_metadata(carrier_selector)

    objects = retained / "objects"
    atomic_bytes(objects / "pr135.models.f24s.raw", base_models)
    atomic_bytes(objects / "cp135_hp3.ihs2_body", hpac_body)
    atomic_bytes(objects / "cp135_semantic.wans_body", semantic_body)
    atomic_bytes(objects / "pass4_pose.cpr1", pass4_carrier)
    atomic_bytes(objects / "pass4_pose.cap1", cap1)
    atomic_bytes(objects / "pass4_pose.f24s_carrier_selector", carrier_selector)
    atomic_bytes(objects / "pass4_pose.f24s_carrier_selector.packed", packed_carrier)
    atomic_bytes(objects / "cp135_hp3.ihs2", hp3)
    atomic_bytes(objects / "cp135_frame0_selector.f0e1", selector)

    models, model_report = cp135._optimal_split_models(
        (hpac_body, semantic_body, packed_carrier),
        variant="t1r1_pass4_standin",  # type: ignore[arg-type]
        representation="packed_dynamic_cap1",
        output=args.output,
        brotli_binary=BROTLI,
    )
    a, b, c = cp135.SPLIT_HEADER.unpack_from(models)
    if c >= PACKED_CAP1_FLAG:
        raise RuntimeError("T1R1 carrier stream does not fit the tagged u15 length")
    models = cp135.SPLIT_HEADER.pack(a, b, c | PACKED_CAP1_FLAG) + models[cp135.SPLIT_HEADER.size :]
    untagged = cp135.SPLIT_HEADER.pack(a, b, c) + models[cp135.SPLIT_HEADER.size :]
    atomic_bytes(objects / "models.split_brotli_untagged_pre_tag", untagged)
    restored = cp135.unpack_split_models(untagged, brotli_binary=BROTLI)
    if restored[:2] != (hpac_body, semantic_body) or unpack_dynamic_cap1_metadata(restored[2]) != carrier_selector:
        raise RuntimeError("tagged T1R1 split models failed independent parse-back")
    residual = cp_parts.residual_payload[4:]
    member = models + residual + token
    archive = deterministic_zip(member)
    repeat = deterministic_zip(member)
    if archive != repeat:
        raise RuntimeError("T1R1 archive repeat is not byte-identical")
    atomic_bytes(objects / "models.split_brotli_t1r1", models)
    atomic_bytes(objects / "residual.compact", residual)
    atomic_bytes(objects / "c1_tokens.hp3.rc64", token)
    atomic_bytes(objects / "p", member)
    atomic_bytes(objects / "archive.zip", archive)
    atomic_bytes(objects / "archive.repeat.zip", repeat)
    runtime = copy_runtime(retained / "adapted_runtime", archive)
    result = {
        "schema": "ddm_t1r1_build_result.v1",
        "complete": True,
        "rehearsal_label": "REHEARSAL_PASS4_STALE_POSE_STANDIN_NOT_A_CANDIDATE",
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "base_cp135": file_record(CP135_ARCHIVE),
        "archive": file_record(objects / "archive.zip"),
        "repeat_archive": file_record(objects / "archive.repeat.zip"),
        "repeat_byte_identical": archive == repeat,
        "delta_bytes_vs_cp135": len(archive) - CP135_BYTES,
        "models": file_record(objects / "models.split_brotli_t1r1"),
        "raw_model_intermediates": {
            "pr135_f24s": file_record(objects / "pr135.models.f24s.raw"),
            "hp3_body": file_record(objects / "cp135_hp3.ihs2_body"),
            "semantic_wans_body": file_record(objects / "cp135_semantic.wans_body"),
            "untagged_split_before_receiver_tag": file_record(objects / "models.split_brotli_untagged_pre_tag"),
        },
        "residual": file_record(objects / "residual.compact"),
        "token_payload": file_record(objects / "c1_tokens.hp3.rc64"),
        "token_identity_receipt": reencode_receipt["token_result"],
        "member": file_record(objects / "p"),
        "hp3": hp3_report,
        "cap1": cap1_report,
        "cap1_encoder_source": file_record(Path(cap_module.__file__)),
        "packed_cap1": pack_report,
        "model_coder_race": model_report,
        "inherited_cp135_selector": file_record(objects / "cp135_frame0_selector.f0e1"),
        "pose_standin": file_record(objects / "pass4_pose.cpr1"),
        "runtime": runtime,
        "terminal_diff_list": [
            "replace pass4_pose.cpr1 with the terminal same-parent pose carrier and rerun all stages"
        ],
        "all_payloads_retained": True,
    }
    atomic_json(retained / "60_BUILD_RESULT.json", result)
    return result


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_receiver_rc64(adapted: Path, retained: Path) -> dict[str, Any]:
    source = adapted / "runtime/entropy/rc64_backend.c"
    library = retained / "receiver_state/rc64_backend.shipped.so"
    temporary = library.with_name(f".{library.name}.{os.getpid()}.tmp")
    library.parent.mkdir(parents=True, exist_ok=True)
    argv = ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(temporary)]
    started = time.time()
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    receipt = {
        "schema": "ddm_t1r1_receiver_rc64_compile.v1",
        "argv": argv,
        "source": file_record(source),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_s": time.time() - started,
    }
    if completed.returncode:
        atomic_json(retained / "receiver_state/RC64_COMPILE_RESULT.json", receipt)
        raise RuntimeError("shipped RC64 backend compilation failed")
    os.replace(temporary, library)
    receipt["library"] = file_record(library)
    atomic_json(retained / "receiver_state/RC64_COMPILE_RESULT.json", receipt)
    return receipt


def receiver_token_parseback(
    *,
    retained: Path,
    adapted: Path,
    parts: Any,
    renderer: Any,
    rc64_module: Any,
) -> dict[str, Any]:
    import torch

    reencode_receipt = json.loads((retained / "50_HP3_REENCODE_RESULT.json").read_text())
    export = json.loads(Path(reencode_receipt["probability_export"]["path"]).read_text())
    identity_record = export["probability_identity"]
    if file_record(Path(identity_record["path"])) != identity_record:
        raise RuntimeError("receiver parse-back probability identity failed custody")
    identity = json.loads(Path(identity_record["path"]).read_text())
    if identity.get("complete_n600") is not True or len(identity.get("frames", ())) != FRAMES:
        raise RuntimeError("receiver parse-back probability identity is incomplete")
    prepared = json.loads((retained / "20_PREPARE_RESULT.json").read_text())
    event_order = np.load(Path(prepared["event_order_payload"]["path"]), mmap_mode="r", allow_pickle=False)
    if event_order.dtype != np.uint8 or event_order.shape != (FRAMES * EVENTS_PER_FRAME,):
        raise RuntimeError("receiver parse-back source events have invalid geometry")

    compile_receipt = compile_receiver_rc64(adapted, retained)
    decoder = rc64_module.NativeDecoder(Path(compile_receipt["library"]["path"]), parts.token_stream)
    group_positions = [
        np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        for mask in renderer.group_masks(torch.device("cpu"))
    ]
    output = retained / "receiver_state/decoded_symbols.shipped_rc64.bin"
    spatial_output = retained / "receiver_state/decoded_spatial_tokens.shipped_rc64.bin"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    spatial_temporary = spatial_output.with_name(f".{spatial_output.name}.{os.getpid()}.tmp")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    started = time.time()
    with temporary.open("wb") as stream, spatial_temporary.open("wb") as spatial_stream:
        for frame, record in enumerate(identity["frames"]):
            code_path = Path(record["codes"]["path"])
            if int(record["frame"]) != frame or file_record(code_path) != record["codes"]:
                raise RuntimeError(f"receiver probability checkpoint failed custody at frame {frame}")
            probabilities = cp135.probability_from_codes(
                np.load(code_path, mmap_mode="r", allow_pickle=False),
                8,
            )
            decoded = decoder.decode(probabilities).astype(np.uint8)
            first = frame * EVENTS_PER_FRAME
            expected = np.asarray(event_order[first : first + EVENTS_PER_FRAME])
            if not np.array_equal(decoded, expected):
                raise RuntimeError(f"shipped RC64 decoder differs at frame {frame}")
            raw = decoded.tobytes()
            spatial_raw = cp135.spatial_frame(decoded, group_positions).tobytes()
            stream.write(raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(raw)
            spatial_digest.update(spatial_raw)
        stream.flush()
        os.fsync(stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    os.replace(temporary, output)
    os.replace(spatial_temporary, spatial_output)
    wall_s = time.time() - started
    if event_digest.hexdigest() != prepared["event_order_raw_sha256"] or spatial_digest.hexdigest() != C1_SHA256:
        raise RuntimeError("shipped RC64 receiver terminal digest differs")
    bit_position = decoder.bit_position
    decoder.close()
    return {
        "schema": "ddm_t1r1_shipped_rc64_parseback.v1",
        "complete": True,
        "events": FRAMES * EVENTS_PER_FRAME,
        "event_order_sha256": event_digest.hexdigest(),
        "spatial_token_sha256": spatial_digest.hexdigest(),
        "decoded_symbols": file_record(output),
        "decoded_spatial_tokens": file_record(spatial_output),
        "decoder_bit_position": bit_position,
        "decode_wall_s": wall_s,
        "fraction_of_30min": wall_s / 1800.0,
        "probability_identity": identity_record,
        "compile_receipt": file_record(retained / "receiver_state/RC64_COMPILE_RESULT.json"),
        "backend_source": compile_receipt["source"],
        "backend_library": compile_receipt["library"],
    }


def parseback(args: argparse.Namespace) -> dict[str, Any]:
    retained = retained_root(args.output)
    build_result = json.loads((retained / "60_BUILD_RESULT.json").read_text())
    adapted = retained / "adapted_runtime"
    archive_path = adapted / "archive.zip"
    data_dir = retained / "inflate_input"
    atomic_bytes(data_dir / "p", (retained / "objects/p").read_bytes())
    source_parseback = cx2_tm1_parseback()
    cp_runtime = cp135.load_runtime(CP135_RUNTIME)
    cp_parts = cp_runtime.read_residual_archive(CP135_ARCHIVE)
    expected_hpac, _ = cp135.step2_ihs2(cp_runtime.read_residual_archive(PR135_ARCHIVE).hpac_blob)
    for name in tuple(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    started = time.time()
    sys.path.insert(0, str(adapted))
    try:
        residual_module = importlib.import_module("runtime.residual_archive")
        f26_module = importlib.import_module("runtime.f26_inflate")
        carrier_module = importlib.import_module("runtime.carrier_repack")
        rc64_module = importlib.import_module("runtime.entropy.rc64")
        weight_module = importlib.import_module("runtime.entropy.renderer_weight_codec")
        parts = residual_module.read_residual_archive(archive_path)
        renderer = f26_module._load_renderer(adapted / "cpr1")
        carrier_blob, selector = carrier_module.split_frame0_selector_carrier(parts.carrier_blob)
        canonical_carrier = carrier_module.materialize_cpr1(carrier_blob, renderer)
        semantic_pose = struct.pack("<II", 40_252, len(canonical_carrier)) + bytes(40_252) + canonical_carrier
        atomic_bytes(retained / "objects/parseback.semantic_pose_adapter_input", semantic_pose)
        _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
        records = weight_module.decode_wans1(parts.semantic_blob)
        receiver_state = retained / "receiver_state"
        atomic_npy(receiver_state / "pose_basis.float32.npy", basis.detach().cpu().numpy())
        atomic_npy(receiver_state / "pose_coefficients.float32.npy", coefficients.detach().cpu().numpy())
        semantic_records = []
        for index, record in enumerate(records):
            path = receiver_state / "semantic" / f"{index:02d}_{record.schema.name}.float32.npy"
            atomic_npy(path, np.ascontiguousarray(record.values, dtype=np.float32))
            semantic_records.append(file_record(path))
        semantic = renderer.SemanticTokenRenderer(96)
        import torch

        semantic.load_state_dict(
            {
                record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
                for record in records
            },
            strict=True,
        )
        inflate_entry = import_from_path("ddm_t1r1_retained_inflate", adapted / "inflate.py")
        inflate_entry._verify_input(data_dir, archive_path)
        shipped_token_parseback = receiver_token_parseback(
            retained=retained,
            adapted=adapted,
            parts=parts,
            renderer=renderer,
            rc64_module=rc64_module,
        )
    finally:
        sys.path.pop(0)
    token_result = json.loads(Path(build_result["token_identity_receipt"]["path"]).read_text())
    if parts.semantic_blob != cp_parts.semantic_blob:
        raise RuntimeError("T1R1 semantic renderer differs from CP135")
    if parts.hpac_blob != expected_hpac or parts.residual_payload != cp_parts.residual_payload:
        raise RuntimeError("T1R1 HP3 probability object or residual table differs from CP135")
    if parts.token_stream != Path(token_result["token_payload"]["path"]).read_bytes():
        raise RuntimeError("T1R1 receiver token stream differs from retained C1 HP3 RC64")
    receiver_backend = file_record(adapted / "runtime/entropy/rc64_backend.c")
    if canonical_carrier != PASS4_CARRIER.read_bytes():
        raise RuntimeError("T1R1 receiver carrier differs from pass-4 CPR1 stand-in")
    if token_result["decoded_spatial_token_sha256"] != C1_SHA256 or token_result["symbol_identity"] is not True:
        raise RuntimeError("T1R1 C1 token decode lacks exact n600 identity")
    direct_state = ps135.decode_carrier(PASS4_CARRIER.read_bytes())
    expected_coefficients = direct_state.codes.astype(np.float32) * direct_state.coefficient_scales[None, :]
    if not np.array_equal(coefficients.numpy(), expected_coefficients):
        raise RuntimeError("F26 receiver coefficients differ from pass-4 CPR1 parse-back")

    negative = retained / "negative_controls"
    corrupt_archive = bytearray(archive_path.read_bytes())
    corrupt_archive[-1] ^= 1
    atomic_bytes(negative / "archive.one_byte_corrupt.zip", bytes(corrupt_archive))
    mismatch_dir = negative / "mismatched_input"
    mismatched_member = bytearray((data_dir / "p").read_bytes())
    mismatched_member[0] ^= 1
    atomic_bytes(mismatch_dir / "p", bytes(mismatched_member))
    refusals = {}
    try:
        inflate_entry._verify_input(data_dir, negative / "archive.one_byte_corrupt.zip")
    except ValueError as error:
        refusals["corrupt_archive"] = str(error)
    try:
        inflate_entry._verify_input(mismatch_dir, archive_path)
    except ValueError as error:
        refusals["mismatched_extracted_payload"] = str(error)
    if set(refusals) != {"corrupt_archive", "mismatched_extracted_payload"}:
        raise RuntimeError("retained inflate entrypoint failed open on a negative control")
    parseback_wall_s = time.time() - started
    result = {
        "schema": "ddm_t1r1_parseback_result.v1",
        "complete": True,
        "rehearsal_label": build_result["rehearsal_label"],
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "archive": file_record(archive_path),
        "source_cx2_tm1_parseback": source_parseback,
        "semantic_identity_to_cp135": True,
        "hp3_probability_object_identity_to_cp135": True,
        "residual_identity_to_cp135": True,
        "pass4_cpr1_identity": True,
        "c1_token_identity": True,
        "token_identity_decoder_backend_behavior_matches_receiver": True,
        "receiver_rc64_backend": receiver_backend,
        "token_identity_decoder_backend_byte_identical_to_receiver": (
            receiver_backend["sha256"] == token_result["source_backend"]["sha256"]
        ),
        "shipped_receiver_token_parseback": shipped_token_parseback,
        "decoded_events": token_result["events"],
        "token_decode_wall_s": shipped_token_parseback["decode_wall_s"],
        "token_decode_fraction_of_30min": shipped_token_parseback["fraction_of_30min"],
        "identity_decoder_token_decode_wall_s": token_result["decode_wall_s"],
        "container_parseback_wall_s": parseback_wall_s,
        "full_cuda_render_performed": False,
        "full_cuda_render_boundary": "not run; scorer-free rehearsal and no scorer/CUDA lane authority",
        "renderer_basis_shape": list(basis.shape),
        "renderer_coefficients_shape": list(coefficients.shape),
        "semantic_state_record_count": len(records),
        "retained_semantic_state_records": semantic_records,
        "retained_pose_basis": file_record(retained / "receiver_state/pose_basis.float32.npy"),
        "retained_pose_coefficients": file_record(retained / "receiver_state/pose_coefficients.float32.npy"),
        "inflate_positive_assertions": "PASS",
        "inflate_negative_control_refusals": refusals,
        "negative_control_payloads": [
            file_record(negative / "archive.one_byte_corrupt.zip"),
            file_record(mismatch_dir / "p"),
        ],
        "receiver_acceptance": True,
    }
    atomic_json(retained / "70_PARSEBACK_RESULT.json", result)
    tree = retained_tree_record(retained)
    atomic_json(retained / "99_TREE_MANIFEST.json", tree)
    return result


def all_stages(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args)
    reencode(args)
    build(args)
    command = [sys.executable, str(Path(__file__).resolve()), "parseback", "--output", str(args.output)]
    run_logged(command, retained_root(args.output) / "logs/70_PARSEBACK.log")
    return json.loads((retained_root(args.output) / "70_PARSEBACK_RESULT.json").read_text())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("prepare", "reencode", "build", "parseback", "all"))
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main() -> None:
    args = parser().parse_args()
    functions = {
        "prepare": prepare,
        "reencode": reencode,
        "build": build,
        "parseback": parseback,
        "all": all_stages,
    }
    result = functions[args.stage](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
