#!/usr/bin/env python3
"""Measure rich lossless estimators on the pinned DX2 token field.

Every coded stream and every decoded output is retained before its size or
identity is admitted.  The run is stage-checkpointed and resumes by validating
completed candidate receipts.  This is a scorer-free research measurement; it
does not edit a receiver or build a contest archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyppmd  # PYPPMD_LGPL_OK:bounded-research-race-retains-wire-payloads-no-package-runtime

AXIS = "[macOS-CPU advisory / scorer-free lossless diagnostic]"
SCHEMA = "ddm_ef1_token_entropy_floor.v1"
SEED = 0
EXPECTED_SOURCE_BYTES = 117_964_800
EXPECTED_SOURCE_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
EXPECTED_ARCHIVE_BYTES = 180_368
EXPECTED_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
EXPECTED_STREAM_BYTES = 113_777
EXPECTED_STREAM_SHA256 = "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"
EXPECTED_CHECKPOINT_BYTES = 3_511
EXPECTED_CHECKPOINT_SHA256 = "c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9"
TARGET_STREAM_BYTES = 71_395
FRAME_SYMBOLS = 384 * 512
DEFAULT_INPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/retained/input")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_ef1_token_entropy_floor/measurement_v1")
MIN_FREE_BYTES = 8 << 30
PPMD_MAGIC = b"EF1P"
PPMD_VARIANT = "H"
PPMD_HEADER_BYTES = 16
FIXED_MTIME_NS = 1_704_067_200_000_000_000
ZPAQ_CANONICAL_TIMESTAMP = b"jDC20000101000000"
ZPAQ_TIMESTAMP_PATTERN = re.compile(rb"jDC[0-9]{14}")
DEFAULT_PPMD_ORDERS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64)
DEFAULT_ZPAQ_METHODS = (1, 2, 3, 4, 5)
DEFAULT_PREFIX_FRAMES = (8, 16, 32, 64, 128, 256, 400, 600)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(path, encoded)


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_receipt(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        receipt = file_receipt(item)
        digest.update(struct.pack("<I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack("<Q", receipt["bytes"]))
        digest.update(bytes.fromhex(receipt["sha256"]))
        total_bytes += receipt["bytes"]
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required file is missing: {path}")
    receipt = file_receipt(path)
    if receipt["bytes"] != expected_bytes or receipt["sha256"] != expected_sha256:
        raise RuntimeError(
            f"pin drift for {path}: got {receipt['bytes']} B {receipt['sha256']}, "
            f"expected {expected_bytes} B {expected_sha256}"
        )
    return receipt


def ensure_expected_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"retained artifact drifted: {path}")


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = output.resolve()
    vertigo = Path("/Volumes/VertigoDataTier/pact").resolve()
    if vertigo not in resolved.parents:
        raise RuntimeError(f"EF1 receipts must be under {vertigo}, got {resolved}")
    stats = os.statvfs(output)
    free = stats.f_bavail * stats.f_frsize
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"storage preflight failed: {free} B free < {MIN_FREE_BYTES} B")
    return {
        "tier": "/Volumes/VertigoDataTier/pact",
        "free_bytes_at_start": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "output_root": str(resolved),
    }


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    atomic_json(
        log_path,
        {
            "argv": command,
            "cwd": str(cwd),
            "elapsed_seconds": elapsed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}; see {log_path}")
    return elapsed


def zpaq_version(binary: str) -> str:
    completed = subprocess.run([binary], text=True, capture_output=True, check=False)
    text = completed.stdout + completed.stderr
    match = re.search(r"zpaq v([0-9.]+)", text)
    if not match:
        raise RuntimeError(f"could not identify zpaq version from {text[:200]!r}")
    return match.group(1)


def canonicalize_zpaq(*, binary: str, raw_archive: Path, canonical_archive: Path, receipt_path: Path) -> dict[str, Any]:
    raw = raw_archive.read_bytes()
    offsets = [match.start() for match in ZPAQ_TIMESTAMP_PATTERN.finditer(raw)]
    if not offsets:
        raise RuntimeError(f"ZPAQ archive has no journal timestamps to canonicalize: {raw_archive}")
    canonical = ZPAQ_TIMESTAMP_PATTERN.sub(ZPAQ_CANONICAL_TIMESTAMP, raw)
    atomic_bytes(canonical_archive, canonical)
    validation_log = receipt_path.with_name(f"{receipt_path.stem}.test.json")
    run_command(
        [binary, "extract", str(canonical_archive), "-test", "-threads", "1"],
        cwd=canonical_archive.parent,
        log_path=validation_log,
    )
    receipt = {
        "raw_archive": file_receipt(raw_archive),
        "canonical_archive": file_receipt(canonical_archive),
        "timestamp_offsets": offsets,
        "timestamp_count": len(offsets),
        "replacement": ZPAQ_CANONICAL_TIMESTAMP.decode("ascii"),
        "validation_log": file_receipt(validation_log),
        "meaning": "generic journal timestamps canonicalized; coded data and model bytes unchanged",
    }
    atomic_json(receipt_path, receipt)
    return receipt


def initial_manifest(args: argparse.Namespace, storage: dict[str, Any]) -> dict[str, Any]:
    zpaq = shutil.which("zpaq")
    if zpaq is None:
        raise RuntimeError("zpaq is required for the context-mixing race")
    return {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "seed": SEED,
        "complete": False,
        "storage_preflight": storage,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "pyppmd": getattr(pyppmd, "__version__", "1.3.1"),
            "zpaq_binary": zpaq,
            "zpaq_version": zpaq_version(zpaq),
        },
        "input_root": str(args.input_root),
        "output_root": str(args.output),
        "pins": {},
        "incumbent": {},
        "ppmd": {},
        "zpaq": {},
        "zpaq_prefix_curve": {},
        "labels": {
            "achieved_total_bytes": "finite-string lossless code size; upper bound on optimal code size",
            "normalized_estimate": "compression-based entropy-rate estimate; not a lower bound",
            "lower_bound": "none established by this experiment",
        },
    }


def load_or_initialize_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    storage = storage_preflight(args.output)
    path = args.output / "MANIFEST.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise RuntimeError(f"refusing incompatible manifest at {path}")
        if manifest.get("input_root") != str(args.input_root):
            raise RuntimeError("input root changed across resume")
        manifest["storage_preflight_resume"] = storage
        return path, manifest
    manifest = initial_manifest(args, storage)
    atomic_json(path, manifest)
    return path, manifest


def stage_pins(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    source = require_file(args.input_root / "dx2_tokens_decoded.u8", EXPECTED_SOURCE_BYTES, EXPECTED_SOURCE_SHA256)
    archive = require_file(args.input_root / "archive.zip", EXPECTED_ARCHIVE_BYTES, EXPECTED_ARCHIVE_SHA256)
    stream = require_file(args.input_root / "dx2_token_stream_rc64.bin", EXPECTED_STREAM_BYTES, EXPECTED_STREAM_SHA256)
    checkpoint = require_file(
        args.input_root / "tokens_cpu_stage_complete.json",
        EXPECTED_CHECKPOINT_BYTES,
        EXPECTED_CHECKPOINT_SHA256,
    )
    checkpoint_payload = json.loads(Path(checkpoint["path"]).read_text(encoding="utf-8"))
    checkpoint_text = json.dumps(checkpoint_payload, sort_keys=True)
    for token in (EXPECTED_ARCHIVE_SHA256, EXPECTED_SOURCE_SHA256, EXPECTED_STREAM_SHA256):
        if token not in checkpoint_text:
            raise RuntimeError(f"checkpoint receipt does not bind expected pin {token}")
    bits_per_position = 8.0 * stream["bytes"] / source["bytes"]
    target_bits_per_position = 8.0 * TARGET_STREAM_BYTES / source["bytes"]
    manifest["pins"] = {
        "source": source,
        "archive": archive,
        "shipped_token_stream": stream,
        "token_checkpoint_receipt": checkpoint,
        "checkpoint_binds_all_pins": True,
    }
    manifest["incumbent"] = {
        "stream_bytes": stream["bytes"],
        "positions": source["bytes"],
        "bits_per_position": bits_per_position,
        "target_stream_bytes": TARGET_STREAM_BYTES,
        "target_bits_per_position": target_bits_per_position,
        "required_cut_bytes": stream["bytes"] - TARGET_STREAM_BYTES,
        "required_density_fraction": target_bits_per_position / bits_per_position,
    }


def ppmd_packet(body: bytes, *, order: int, memory_bytes: int, source_bytes: int) -> bytes:
    if memory_bytes <= 0 or memory_bytes & (memory_bytes - 1):
        raise ValueError("PPMd memory must be a positive power of two")
    log2_memory = memory_bytes.bit_length() - 1
    return (
        PPMD_MAGIC
        + PPMD_VARIANT.encode("ascii")
        + bytes((order, log2_memory, 0))
        + struct.pack("<Q", source_bytes)
        + body
    )


def decode_ppmd_packet(path: Path) -> bytes:
    packet = path.read_bytes()
    if len(packet) < PPMD_HEADER_BYTES or packet[:4] != PPMD_MAGIC:
        raise RuntimeError(f"invalid EF1 PPMd packet: {path}")
    variant = packet[4:5].decode("ascii")
    order = packet[5]
    memory_bytes = 1 << packet[6]
    source_bytes = struct.unpack_from("<Q", packet, 8)[0]
    if variant == "H":
        decoder = pyppmd.Ppmd7Decoder(max_order=order, mem_size=memory_bytes)
        decoded = decoder.decode(packet[PPMD_HEADER_BYTES:], source_bytes)
    else:
        decoded = pyppmd.decompress(packet[PPMD_HEADER_BYTES:], max_order=order, mem_size=memory_bytes, variant=variant)
    if not isinstance(decoded, bytes) or len(decoded) != source_bytes:
        raise RuntimeError(f"PPMd decoded length mismatch for {path}")
    return decoded


def valid_row(row: dict[str, Any], source_sha256: str, source_bytes: int) -> bool:
    try:
        if not row.get("complete") or not row.get("exact_decode"):
            return False
        if row.get("source_sha256") != source_sha256 or row.get("source_bytes") != source_bytes:
            return False
        for key in ("payload", "payload_repeat", "decoded"):
            receipt = row[key]
            path = Path(receipt["path"])
            ensure_expected_file(path, receipt["bytes"], receipt["sha256"])
        return row["payload"]["sha256"] == row["payload_repeat"]["sha256"]
    except (KeyError, OSError, TypeError, RuntimeError):
        return False


def run_ppmd(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> None:
    source_path = Path(manifest["pins"]["source"]["path"])
    source_sha = manifest["pins"]["source"]["sha256"]
    source_bytes = source_path.read_bytes()
    memory_bytes = args.ppmd_memory_mib << 20
    ppmd_root = args.output / "retained" / "ppmd"
    ppmd_rows = manifest.setdefault("ppmd", {})
    ppmd_rows["estimator"] = f"pyppmd {getattr(pyppmd, '__version__', '1.3.1')} PPMd7 variant H adaptive arithmetic"
    ppmd_rows["transmitted_model_bytes"] = 0
    ppmd_rows["parameter_header_bytes"] = PPMD_HEADER_BYTES
    ppmd_rows["memory_bytes"] = memory_bytes
    candidates = ppmd_rows.setdefault("candidates", {})
    for order in args.ppmd_orders:
        candidate_id = f"order_{order:02d}_mem_{args.ppmd_memory_mib:04d}mib"
        old = candidates.get(candidate_id, {})
        if valid_row(old, source_sha, EXPECTED_SOURCE_BYTES):
            continue
        candidate = ppmd_root / candidate_id
        candidate.mkdir(parents=True, exist_ok=True)
        body_path = candidate / "body.ppmd"
        packet_path = candidate / "stream.ef1p"
        body_repeat_path = candidate / "body.repeat.ppmd"
        packet_repeat_path = candidate / "stream.repeat.ef1p"
        decoded_path = candidate / "decoded.u8"
        started = time.perf_counter()
        try:
            if not packet_path.exists():
                body = pyppmd.compress(source_bytes, max_order=order, mem_size=memory_bytes, variant=PPMD_VARIANT)
                atomic_bytes(body_path, body)
                packet = ppmd_packet(body, order=order, memory_bytes=memory_bytes, source_bytes=EXPECTED_SOURCE_BYTES)
                atomic_bytes(packet_path, packet)
            encode_seconds = time.perf_counter() - started
            repeat_started = time.perf_counter()
            if not packet_repeat_path.exists():
                body_repeat = pyppmd.compress(
                    source_bytes, max_order=order, mem_size=memory_bytes, variant=PPMD_VARIANT
                )
                atomic_bytes(body_repeat_path, body_repeat)
                packet_repeat = ppmd_packet(
                    body_repeat,
                    order=order,
                    memory_bytes=memory_bytes,
                    source_bytes=EXPECTED_SOURCE_BYTES,
                )
                atomic_bytes(packet_repeat_path, packet_repeat)
            repeat_encode_seconds = time.perf_counter() - repeat_started
            if sha256_file(packet_path) != sha256_file(packet_repeat_path):
                raise RuntimeError(f"PPMd repeat mismatch for order {order}")
            decode_started = time.perf_counter()
            if not decoded_path.exists():
                decoded = decode_ppmd_packet(packet_path)
                atomic_bytes(decoded_path, decoded)
            decode_seconds = time.perf_counter() - decode_started
            if sha256_file(decoded_path) != source_sha:
                raise RuntimeError(f"PPMd exact inversion failed for order {order}")
            row = {
                "complete": True,
                "estimator": ppmd_rows["estimator"],
                "max_order": order,
                "memory_bytes": memory_bytes,
                "source_bytes": EXPECTED_SOURCE_BYTES,
                "source_sha256": source_sha,
                "transmitted_model_bytes": 0,
                "parameter_header_bytes": PPMD_HEADER_BYTES,
                "body": file_receipt(body_path),
                "body_repeat": file_receipt(body_repeat_path),
                "payload": file_receipt(packet_path),
                "payload_repeat": file_receipt(packet_repeat_path),
                "decoded": file_receipt(decoded_path),
                "exact_decode": True,
                "deterministic_repeat": True,
                "encode_seconds": encode_seconds,
                "repeat_encode_seconds": repeat_encode_seconds,
                "decode_seconds": decode_seconds,
            }
            row["achieved_bits_per_position"] = 8.0 * row["payload"]["bytes"] / EXPECTED_SOURCE_BYTES
            row["delta_vs_incumbent_bytes"] = row["payload"]["bytes"] - EXPECTED_STREAM_BYTES
            row["delta_vs_target_bytes"] = row["payload"]["bytes"] - TARGET_STREAM_BYTES
            candidates[candidate_id] = row
        except Exception as error:
            error_path = candidate / "ERROR.json"
            atomic_json(
                error_path,
                {
                    "candidate": candidate_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            candidates[candidate_id] = {
                "complete": False,
                "candidate_invalid": True,
                "error": file_receipt(error_path),
            }
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)


def zpaq_add(*, binary: str, method: int, source: Path, archive: Path, log_path: Path) -> float:
    if archive.exists():
        validation = subprocess.run(
            [binary, "extract", str(archive), "-test", "-threads", "1"],
            cwd=archive.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if validation.returncode == 0:
            return 0.0
        suffix = 1
        while archive.with_name(f"{archive.name}.interrupted.{suffix}").exists():
            suffix += 1
        interrupted = archive.with_name(f"{archive.name}.interrupted.{suffix}")
        os.replace(archive, interrupted)
        atomic_json(
            archive.with_name(f"{interrupted.name}.receipt.json"),
            {
                "disposition": "retained interrupted ZPAQ output; safe resume creates a fresh archive",
                "artifact": file_receipt(interrupted),
                "validation_returncode": validation.returncode,
                "validation_stdout": validation.stdout,
                "validation_stderr": validation.stderr,
            },
        )
    command = [
        binary,
        "add",
        str(archive),
        source.name,
        "-method",
        str(method),
        "-threads",
        "1",
        "-noattributes",
    ]
    return run_command(command, cwd=source.parent, log_path=log_path)


def zpaq_extract(
    *,
    binary: str,
    archive: Path,
    output: Path,
    log_path: Path,
    cwd: Path,
    expected_bytes: int,
    expected_sha256: str,
    archived_name: str,
) -> float:
    if output.exists():
        if output.stat().st_size == expected_bytes and sha256_file(output) == expected_sha256:
            return 0.0
        suffix = 1
        while output.with_name(f"{output.name}.interrupted.{suffix}").exists():
            suffix += 1
        interrupted = output.with_name(f"{output.name}.interrupted.{suffix}")
        os.replace(output, interrupted)
        atomic_json(
            output.with_name(f"{interrupted.name}.receipt.json"),
            {
                "disposition": "retained incomplete decoded output; safe resume extracts afresh",
                "artifact": file_receipt(interrupted),
            },
        )
    partial = output.with_name(f".{output.name}.extracting")
    extracted = partial / archived_name
    if (
        partial.exists()
        and partial.is_dir()
        and extracted.is_file()
        and extracted.stat().st_size == expected_bytes
        and sha256_file(extracted) == expected_sha256
    ):
        os.replace(extracted, output)
        partial.rmdir()
        return 0.0
    if partial.exists():
        suffix = 1
        while output.with_name(f"{output.name}.interrupted_extract.{suffix}").exists():
            suffix += 1
        interrupted = output.with_name(f"{output.name}.interrupted_extract.{suffix}")
        os.replace(partial, interrupted)
        atomic_json(
            output.with_name(f"{interrupted.name}.receipt.json"),
            {
                "disposition": "retained interrupted extraction scratch; safe resume extracts afresh",
                "artifact": tree_receipt(interrupted) if interrupted.is_dir() else file_receipt(interrupted),
            },
        )
    command = [binary, "extract", str(archive), "-to", partial.name, "-force", "-threads", "1"]
    elapsed = run_command(command, cwd=cwd, log_path=log_path)
    if not extracted.is_file():
        raise RuntimeError(f"ZPAQ did not extract expected member {archived_name} from {archive}")
    if extracted.stat().st_size != expected_bytes or sha256_file(extracted) != expected_sha256:
        raise RuntimeError(f"ZPAQ extracted output mismatch for {archive}")
    os.replace(extracted, output)
    partial.rmdir()
    return elapsed


def valid_zpaq_row(row: dict[str, Any], source_sha256: str, source_bytes: int) -> bool:
    return (
        valid_row(row, source_sha256, source_bytes)
        and row.get("model_description_control", {}).get("bytes") is not None
    )


def prepare_empty_control(args: argparse.Namespace) -> Path:
    path = args.output / "retained" / "zpaq_controls" / "empty.u8"
    if not path.exists():
        atomic_bytes(path, b"")
        os.utime(path, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))
    elif path.stat().st_size != 0:
        raise RuntimeError(f"empty control drifted: {path}")
    return path


def zpaq_control(
    args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], method: int
) -> dict[str, Any]:
    binary = manifest["host"]["zpaq_binary"]
    empty = prepare_empty_control(args)
    root = args.output / "retained" / "zpaq_controls" / f"method_{method}"
    root.mkdir(parents=True, exist_ok=True)
    primary_raw = root / "empty.raw.zpaq"
    repeat_raw = root / "empty.repeat.raw.zpaq"
    primary = root / "empty.zpaq"
    repeat = root / "empty.repeat.zpaq"
    decoded = root / "empty.decoded.u8"
    if primary.exists() and not primary_raw.exists():
        os.replace(primary, primary_raw)
    if repeat.exists() and not repeat_raw.exists():
        os.replace(repeat, repeat_raw)
    zpaq_add(binary=binary, method=method, source=empty, archive=primary_raw, log_path=root / "add.json")
    zpaq_add(binary=binary, method=method, source=empty, archive=repeat_raw, log_path=root / "add.repeat.json")
    primary_canonicalization = canonicalize_zpaq(
        binary=binary,
        raw_archive=primary_raw,
        canonical_archive=primary,
        receipt_path=root / "canonicalize.json",
    )
    repeat_canonicalization = canonicalize_zpaq(
        binary=binary,
        raw_archive=repeat_raw,
        canonical_archive=repeat,
        receipt_path=root / "canonicalize.repeat.json",
    )
    if sha256_file(primary) != sha256_file(repeat):
        raise RuntimeError(f"ZPAQ empty-control repeat mismatch for method {method}")
    zpaq_extract(
        binary=binary,
        archive=primary,
        output=decoded,
        log_path=root / "extract.json",
        cwd=root,
        expected_bytes=0,
        expected_sha256=hashlib.sha256(b"").hexdigest(),
        archived_name=empty.name,
    )
    if decoded.stat().st_size != 0 or sha256_file(decoded) != hashlib.sha256(b"").hexdigest():
        raise RuntimeError(f"ZPAQ empty-control decode mismatch for method {method}")
    control = file_receipt(primary)
    control["raw"] = file_receipt(primary_raw)
    control["repeat"] = file_receipt(repeat)
    control["repeat_raw"] = file_receipt(repeat_raw)
    control["decoded"] = file_receipt(decoded)
    control["canonicalization"] = primary_canonicalization
    control["repeat_canonicalization"] = repeat_canonicalization
    control["meaning"] = (
        "measured fixed archive/model-description/framing control for a zero-byte file; "
        "it is not subtracted from achieved candidate bytes"
    )
    manifest.setdefault("zpaq", {}).setdefault("controls", {})[str(method)] = control
    atomic_json(manifest_path, manifest)
    return control


def run_zpaq_candidate(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    method: int,
    source: Path,
    candidate_root: Path,
    row_store: dict[str, Any],
    row_id: str,
) -> dict[str, Any]:
    source_receipt = file_receipt(source)
    old = row_store.get(row_id, {})
    if valid_zpaq_row(old, source_receipt["sha256"], source_receipt["bytes"]):
        return old
    binary = manifest["host"]["zpaq_binary"]
    control = zpaq_control(args, manifest_path, manifest, method)
    candidate_root.mkdir(parents=True, exist_ok=True)
    primary_raw = candidate_root / "stream.raw.zpaq"
    repeat_raw = candidate_root / "stream.repeat.raw.zpaq"
    primary = candidate_root / "stream.zpaq"
    repeat = candidate_root / "stream.repeat.zpaq"
    decoded = candidate_root / "decoded.u8"
    if primary.exists() and not primary_raw.exists():
        os.replace(primary, primary_raw)
    if repeat.exists() and not repeat_raw.exists():
        os.replace(repeat, repeat_raw)
    encode_seconds = zpaq_add(
        binary=binary,
        method=method,
        source=source,
        archive=primary_raw,
        log_path=candidate_root / "add.json",
    )
    repeat_encode_seconds = zpaq_add(
        binary=binary,
        method=method,
        source=source,
        archive=repeat_raw,
        log_path=candidate_root / "add.repeat.json",
    )
    primary_canonicalization = canonicalize_zpaq(
        binary=binary,
        raw_archive=primary_raw,
        canonical_archive=primary,
        receipt_path=candidate_root / "canonicalize.json",
    )
    repeat_canonicalization = canonicalize_zpaq(
        binary=binary,
        raw_archive=repeat_raw,
        canonical_archive=repeat,
        receipt_path=candidate_root / "canonicalize.repeat.json",
    )
    if sha256_file(primary) != sha256_file(repeat):
        raise RuntimeError(f"ZPAQ repeat mismatch for {row_id}")
    decode_seconds = zpaq_extract(
        binary=binary,
        archive=primary,
        output=decoded,
        log_path=candidate_root / "extract.json",
        cwd=candidate_root,
        expected_bytes=source_receipt["bytes"],
        expected_sha256=source_receipt["sha256"],
        archived_name=source.name,
    )
    if sha256_file(decoded) != source_receipt["sha256"]:
        raise RuntimeError(f"ZPAQ exact inversion failed for {row_id}")
    row = {
        "complete": True,
        "estimator": f"ZPAQ v{manifest['host']['zpaq_version']} method {method}, single-thread context mixing",
        "method": method,
        "source_bytes": source_receipt["bytes"],
        "source_sha256": source_receipt["sha256"],
        "transmitted_learned_model_bytes": 0,
        "model_description_control": control,
        "raw_payload": file_receipt(primary_raw),
        "raw_payload_repeat": file_receipt(repeat_raw),
        "canonicalization": primary_canonicalization,
        "repeat_canonicalization": repeat_canonicalization,
        "payload": file_receipt(primary),
        "payload_repeat": file_receipt(repeat),
        "decoded": file_receipt(decoded),
        "exact_decode": True,
        "deterministic_repeat": True,
        "encode_seconds": encode_seconds,
        "repeat_encode_seconds": repeat_encode_seconds,
        "decode_seconds": decode_seconds,
    }
    row["achieved_bits_per_position"] = 8.0 * row["payload"]["bytes"] / row["source_bytes"]
    row["normalized_estimate_excluding_fixed_control_bits_per_position"] = (
        8.0 * max(0, row["payload"]["bytes"] - control["bytes"]) / row["source_bytes"]
    )
    if row["source_bytes"] == EXPECTED_SOURCE_BYTES:
        row["delta_vs_incumbent_bytes"] = row["payload"]["bytes"] - EXPECTED_STREAM_BYTES
        row["delta_vs_target_bytes"] = row["payload"]["bytes"] - TARGET_STREAM_BYTES
    row_store[row_id] = row
    atomic_json(manifest_path, manifest)
    return row


def run_zpaq_full(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> None:
    source = Path(manifest["pins"]["source"]["path"])
    zpaq_rows = manifest.setdefault("zpaq", {})
    candidates = zpaq_rows.setdefault("candidates", {})
    for method in args.zpaq_methods:
        row_id = f"method_{method}"
        run_zpaq_candidate(
            args=args,
            manifest_path=manifest_path,
            manifest=manifest,
            method=method,
            source=source,
            candidate_root=args.output / "retained" / "zpaq" / row_id,
            row_store=candidates,
            row_id=row_id,
        )


def materialize_prefix(source: Path, output: Path, symbols: int) -> dict[str, Any]:
    if output.exists():
        if output.stat().st_size != symbols:
            raise RuntimeError(f"prefix length drifted: {output}")
        return file_receipt(output)
    with source.open("rb") as handle:
        payload = handle.read(symbols)
    if len(payload) != symbols:
        raise RuntimeError(f"source ended before {symbols} symbols")
    atomic_bytes(output, payload)
    os.utime(output, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))
    return file_receipt(output)


def run_zpaq_curve(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> None:
    source = Path(manifest["pins"]["source"]["path"])
    curve = manifest.setdefault("zpaq_prefix_curve", {})
    curve["estimator"] = (
        f"ZPAQ v{manifest['host']['zpaq_version']} method {args.curve_method}, single-thread context mixing"
    )
    curve["method"] = args.curve_method
    rows = curve.setdefault("rows", {})
    prefix_root = args.output / "retained" / "zpaq_prefix_curve"
    for frames in args.prefix_frames:
        if frames <= 0 or frames > 600:
            raise ValueError(f"prefix frame count out of range: {frames}")
        symbols = frames * FRAME_SYMBOLS
        row_id = f"frames_{frames:03d}"
        candidate_root = prefix_root / row_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        if frames == 600:
            full_row = manifest.get("zpaq", {}).get("candidates", {}).get(f"method_{args.curve_method}")
            if not full_row or not full_row.get("complete"):
                raise RuntimeError("full-field ZPAQ row must complete before the 600-frame curve point")
            row = json.loads(json.dumps(full_row))
            source_receipt = file_receipt(source)
            row["reused_full_candidate"] = True
            rows[row_id] = row
        else:
            prefix = candidate_root / "source.u8"
            source_receipt = materialize_prefix(source, prefix, symbols)
            row = run_zpaq_candidate(
                args=args,
                manifest_path=manifest_path,
                manifest=manifest,
                method=args.curve_method,
                source=prefix,
                candidate_root=candidate_root,
                row_store=rows,
                row_id=row_id,
            )
        row["frames"] = frames
        row["positions"] = symbols
        row["prefix_source"] = source_receipt
        atomic_json(manifest_path, manifest)
    ordered = sorted(rows.values(), key=lambda row: row["positions"])
    previous: dict[str, Any] | None = None
    for row in ordered:
        if previous is not None:
            delta_bits = 8.0 * (row["payload"]["bytes"] - previous["payload"]["bytes"])
            delta_positions = row["positions"] - previous["positions"]
            row["marginal_bits_per_position_since_previous_prefix"] = delta_bits / delta_positions
        previous = row
    curve["ordered_row_ids"] = [f"frames_{row['frames']:03d}" for row in ordered]
    atomic_json(manifest_path, manifest)


def finish(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> None:
    ppmd_valid = [row for row in manifest.get("ppmd", {}).get("candidates", {}).values() if row.get("complete")]
    zpaq_valid = [row for row in manifest.get("zpaq", {}).get("candidates", {}).values() if row.get("complete")]
    curve_valid = [row for row in manifest.get("zpaq_prefix_curve", {}).get("rows", {}).values() if row.get("complete")]
    if not ppmd_valid or not zpaq_valid or not curve_valid:
        raise RuntimeError("cannot finalize before all three estimator surfaces have retained rows")
    measured_orders = {row["max_order"] for row in ppmd_valid}
    measured_methods = {row["method"] for row in zpaq_valid}
    measured_prefixes = {row["frames"] for row in curve_valid}
    if measured_orders != set(DEFAULT_PPMD_ORDERS):
        raise RuntimeError(f"PPMd order curve incomplete: {sorted(measured_orders)}")
    if measured_methods != set(DEFAULT_ZPAQ_METHODS):
        raise RuntimeError(f"ZPAQ method curve incomplete: {sorted(measured_methods)}")
    if measured_prefixes != set(DEFAULT_PREFIX_FRAMES):
        raise RuntimeError(f"ZPAQ prefix curve incomplete: {sorted(measured_prefixes)}")
    all_full = ppmd_valid + zpaq_valid
    best = min(all_full, key=lambda row: row["payload"]["bytes"])
    manifest["summary"] = {
        "best_estimator": best["estimator"],
        "best_payload": best["payload"],
        "best_bits_per_position": best["achieved_bits_per_position"],
        "best_delta_vs_incumbent_bytes": best["payload"]["bytes"] - EXPECTED_STREAM_BYTES,
        "best_delta_vs_target_bytes": best["payload"]["bytes"] - TARGET_STREAM_BYTES,
        "target_reached": best["payload"]["bytes"] <= TARGET_STREAM_BYTES,
        "incumbent_beaten": best["payload"]["bytes"] < EXPECTED_STREAM_BYTES,
        "lower_bound_established": False,
        "lower_bound_statement": (
            "No estimator code size or normalized compression curve is an information-theoretic lower bound."
        ),
    }
    manifest["complete"] = True
    manifest["authoritative_rebuild_command"] = (
        f"{sys.executable} experiments/ddm_ef1_token_entropy_floor.py "
        f"--input-root {args.input_root} --output {args.output}"
    )
    atomic_json(manifest_path, manifest)
    inventory: list[dict[str, Any]] = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "INVENTORY.json":
            inventory.append(file_receipt(path))
    total_bytes = sum(row["bytes"] for row in inventory)
    atomic_json(
        args.output / "INVENTORY.json",
        {
            "schema": "ddm_ef1_retention_inventory.v1",
            "axis": AXIS,
            "score_claim": False,
            "root": str(args.output),
            "artifact_count": len(inventory),
            "total_bytes": total_bytes,
            "artifacts": inventory,
            "rebuild_command": manifest["authoritative_rebuild_command"],
            "cleanup_disposition": (
                "retain in place on the preferred Vertigo SSD tier; no local scratch or APDataStore payloads created"
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("pins", "ppmd", "zpaq", "curve", "finish"),
        default=("pins", "ppmd", "zpaq", "curve", "finish"),
    )
    parser.add_argument("--ppmd-orders", nargs="+", type=int, default=DEFAULT_PPMD_ORDERS)
    parser.add_argument("--ppmd-memory-mib", type=int, default=256)
    parser.add_argument("--zpaq-methods", nargs="+", type=int, default=DEFAULT_ZPAQ_METHODS)
    parser.add_argument("--curve-method", type=int, default=5)
    parser.add_argument("--prefix-frames", nargs="+", type=int, default=DEFAULT_PREFIX_FRAMES)
    args = parser.parse_args()
    if args.ppmd_memory_mib <= 0 or args.ppmd_memory_mib & (args.ppmd_memory_mib - 1):
        parser.error("--ppmd-memory-mib must be a positive power of two")
    if any(order <= 0 or order > 64 for order in args.ppmd_orders):
        parser.error("PPMd orders must be in 1..64")
    if any(method < 1 or method > 5 for method in args.zpaq_methods) or not 1 <= args.curve_method <= 5:
        parser.error("ZPAQ methods must be in 1..5")
    return args


def main() -> int:
    args = parse_args()
    manifest_path, manifest = load_or_initialize_manifest(args)
    if "pins" in args.stages or not manifest.get("pins"):
        stage_pins(args, manifest)
        atomic_json(manifest_path, manifest)
    if "ppmd" in args.stages:
        run_ppmd(args, manifest_path, manifest)
    if "zpaq" in args.stages:
        run_zpaq_full(args, manifest_path, manifest)
    if "curve" in args.stages:
        run_zpaq_curve(args, manifest_path, manifest)
    if "finish" in args.stages:
        finish(args, manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
