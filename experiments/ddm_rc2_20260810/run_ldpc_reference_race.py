#!/usr/bin/env python3
"""Resumable retained n600 LDPC-syndrome race on the PR130 HPAC flip field.

The receiver legitimately owns each group's HPAC logits after decoding the
causal prefix.  The Rust coder transmits an LDPC syndrome for the top-class hit
field, exact raw fallback bits only for groups where deterministic min-sum BP
does not reconstruct the encoder object, and an arithmetic-coded miss-class
stream.  Every stage packet, attempted syndrome, decoded output, log, and final
candidate packet is retained on APDataStore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

AXIS = "[macOS-CPU advisory, scorer-free]"
SCORE_CLAIM = False
SEED = 0
SCHEMA = "ddm_rc2_ldpc_reference_race.v2"
SET_MAGIC = b"RC2LSET1"
REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_MANIFEST = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json")
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_rc2_20260810/ldpc_reference_tiefirst")
RUST_MANIFEST = Path(__file__).with_name("ldpc_native") / "Cargo.toml"
RUST_BINARY = Path(__file__).with_name("ldpc_native") / "target/release/ddm-rc2-ldpc-native"
TOKEN_INCUMBENT_BYTES = 114_860
TOKEN_MEMORYLESS_BOUND_BYTES = 114_852
MIN_FREE_BYTES = 5 << 30


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
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def storage_preflight(output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(output)
    free = stats.f_bavail * stats.f_frsize
    total = stats.f_blocks * stats.f_frsize
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"storage preflight: {free} free bytes < {MIN_FREE_BYTES} required")
    return {"free_bytes": free, "total_bytes": total, "required_free_bytes": MIN_FREE_BYTES}


def npy_data_offset(path: Path) -> int:
    with path.open("rb") as handle:
        prefix = handle.read(12)
    if prefix[:6] != b"\x93NUMPY":
        raise RuntimeError(f"not NPY: {path}")
    major = prefix[6]
    if major == 1:
        return 10 + struct.unpack_from("<H", prefix, 8)[0]
    return 12 + struct.unpack_from("<I", prefix, 8)[0]


def hash_npy_payloads(rows: list[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for row in rows:
        path = Path(row["symbols_path"])
        with path.open("rb") as handle:
            handle.seek(npy_data_offset(path))
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
    return digest.hexdigest(), total


def parse_variant(text: str) -> tuple[int, int, int]:
    try:
        alpha, degree, iterations = (int(part) for part in text.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("variant must be ALPHA_MILLI:DEGREE:ITERATIONS") from error
    if alpha <= 0 or not 1 <= degree <= 16 or not 1 <= iterations <= 255:
        raise argparse.ArgumentTypeError("variant values outside supported positive ranges")
    return alpha, degree, iterations


def variant_name(variant: tuple[int, int, int]) -> str:
    alpha, degree, iterations = variant
    return f"alpha_{alpha:05d}_degree_{degree:02d}_iter_{iterations:03d}"


def run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    transcript = (
        b"COMMAND\n" + json.dumps(command).encode() + b"\nSTDOUT\n" + process.stdout + b"\nSTDERR\n" + process.stderr
    )
    atomic_bytes(log_path, transcript)
    if process.returncode != 0:
        raise RuntimeError(f"command failed rc={process.returncode}; retained transcript: {log_path}")
    return {
        "command": command,
        "returncode": process.returncode,
        "wall_s": elapsed,
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size,
        "log_sha256": sha256_file(log_path),
    }


def valid_stage(stage: dict[str, Any], source_sha: str, binary_sha: str) -> bool:
    try:
        if (
            stage.get("complete") is not True
            or stage["source_sha256"] != source_sha
            or stage["native_binary_sha256"] != binary_sha
        ):
            return False
        for key in (
            "packet",
            "attempted_syndromes",
            "decoded",
            "encode_metrics",
            "decode_metrics",
            "encode_log",
            "decode_log",
        ):
            item = stage[key]
            path = Path(item["path"])
            if not (path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"]):
                return False
        return stage["decoded"]["sha256"] == source_sha
    except (KeyError, OSError, TypeError):
        return False


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def stage_source_sha(row: dict[str, Any]) -> str:
    path = Path(row["symbols_path"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(npy_data_offset(path))
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage(
    binary: Path,
    binary_sha: str,
    candidate_dir: Path,
    row: dict[str, Any],
    variant: tuple[int, int, int],
) -> dict[str, Any]:
    alpha, degree, iterations = variant
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    stage_dir = candidate_dir / "stages" / f"frames_{start:04d}_{end:04d}"
    source_sha = stage_source_sha(row)
    encode_command = [
        str(binary),
        "encode",
        "--symbols",
        row["symbols_path"],
        "--codes",
        row["codes_path"],
        "--output",
        str(stage_dir),
        "--alpha-milli",
        str(alpha),
        "--degree",
        str(degree),
        "--iterations",
        str(iterations),
        "--group-offset",
        str(start * 190),
    ]
    encode_run = run_logged(encode_command, stage_dir / "encode_run.log")
    packet = stage_dir / "chunk_packet.bin"
    decoded = stage_dir / "decoded_symbols.bin"
    decode_command = [
        str(binary),
        "decode",
        "--symbols",
        row["symbols_path"],
        "--codes",
        row["codes_path"],
        "--packet",
        str(packet),
        "--output",
        str(decoded),
    ]
    decode_run = run_logged(decode_command, stage_dir / "decode_run.log")
    decoded_sha = sha256_file(decoded)
    if decoded_sha != source_sha:
        raise RuntimeError(f"native LDPC stage decode differs for frames {start}:{end}")
    encode_metrics = json.loads((stage_dir / "encode_metrics.json").read_text())
    decode_metrics = json.loads((stage_dir / "decode_metrics.json").read_text())
    return {
        "complete": True,
        "start_frame": start,
        "end_frame": end,
        "source_sha256": source_sha,
        "native_binary_sha256": binary_sha,
        "packet": artifact(packet),
        "attempted_syndromes": artifact(stage_dir / "attempted_syndromes.bin"),
        "decoded": artifact(decoded),
        "encode_metrics": artifact(stage_dir / "encode_metrics.json"),
        "decode_metrics": artifact(stage_dir / "decode_metrics.json"),
        "encode_log": artifact(stage_dir / "encode_run.log"),
        "decode_log": artifact(stage_dir / "decode_run.log"),
        "encode": encode_metrics,
        "decode": decode_metrics,
        "encode_invocation": encode_run,
        "decode_invocation": decode_run,
    }


def assemble_candidate(candidate_dir: Path, stages: list[dict[str, Any]]) -> dict[str, Any]:
    packet_path = candidate_dir / "tokens.ldpc"
    temporary = packet_path.with_name(f".{packet_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(SET_MAGIC)
        output.write(struct.pack("<I", len(stages)))
        for stage in stages:
            packet = Path(stage["packet"]["path"])
            output.write(struct.pack("<I", packet.stat().st_size))
            with packet.open("rb") as source:
                shutil.copyfileobj(source, output, 1 << 20)
    os.replace(temporary, packet_path)

    decoded_path = candidate_dir / "decoded_symbols_n600.bin"
    temporary = decoded_path.with_name(f".{decoded_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        for stage in stages:
            with Path(stage["decoded"]["path"]).open("rb") as source:
                shutil.copyfileobj(source, output, 1 << 20)
    os.replace(temporary, decoded_path)
    return {"packet": artifact(packet_path), "decoded": artifact(decoded_path)}


def build_binary(output: Path) -> dict[str, Any]:
    build = run_logged(
        ["cargo", "build", "--release", "--manifest-path", str(RUST_MANIFEST)],
        output / "native_build.log",
    )
    if not RUST_BINARY.is_file():
        raise RuntimeError("cargo reported success without the rc2 native binary")
    retained = output / "native_binary" / "ddm-rc2-ldpc-native"
    atomic_bytes(retained, RUST_BINARY.read_bytes())
    retained.chmod(0o755)
    return {"build": build, "working_tree_binary": artifact(RUST_BINARY), "retained_binary": artifact(retained)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    storage = storage_preflight(args.output)
    chunk_manifest = json.loads(CHUNK_MANIFEST.read_text())
    rows = sorted(chunk_manifest["chunks"], key=lambda row: int(row["start_frame"]))
    if not chunk_manifest.get("complete") or rows[0]["start_frame"] != 0 or rows[-1]["end_frame"] != 600:
        raise RuntimeError("DT1 retained chunk manifest is not a complete n600 source")
    source_sha, source_bytes = hash_npy_payloads(rows)
    binary_receipt = build_binary(args.output)
    binary = Path(binary_receipt["retained_binary"]["path"])
    binary_sha = binary_receipt["retained_binary"]["sha256"]
    manifest_path = args.output / "ldpc_reference_manifest.json"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "complete": False,
        "seed": SEED,
        "storage_preflight": storage,
        "source": {
            "chunk_manifest": str(CHUNK_MANIFEST),
            "chunk_manifest_sha256": sha256_file(CHUNK_MANIFEST),
            "symbols": source_bytes,
            "symbols_sha256": source_sha,
            "receiver_side_information": "PR130 HPAC per-group logits, regenerated after causal prefix decode",
        },
        "incumbent": {
            "name": "retained ANS under the existing HPAC model",
            "bytes": TOKEN_INCUMBENT_BYTES,
            "memoryless_or_model_bound_bytes": TOKEN_MEMORYLESS_BOUND_BYTES,
        },
        "native": binary_receipt,
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "rule_118": {
            "free_receiver_code": "generic deterministic QC-like sparse graph, min-sum BP, arithmetic decoder",
            "counted_payload": "fallback flags, selected syndromes/raw exact fallbacks, miss arithmetic stream, all framing",
            "research_only_not_counted": "attempted syndromes and decoded parity outputs retained outside tokens.ldpc",
        },
        "wire_semantics": {
            "top_class_tie_break": "first equal maximum, matching PR130 NumPy/Torch argmax",
            "stage_binary_identity_required_for_resume": True,
        },
        "section_applicability": {
            "tokens": "APPLICABLE: HPAC logits are decoder-owned correlated side information",
            "semantic": "NOT_APPLICABLE: unchanged raw weights have no decoder-owned correlated source",
            "pose": "NOT_APPLICABLE: unchanged carrier bytes have no decoder-owned correlated source",
            "hpac": "NOT_APPLICABLE: unchanged model bytes have no decoder-owned correlated source",
        },
        "candidates": {},
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("schema") == SCHEMA and prior.get("source", {}).get("symbols_sha256") == source_sha:
            manifest["candidates"] = prior.get("candidates", {})

    for variant in args.variants:
        name = variant_name(variant)
        candidate_dir = args.output / name
        candidate = manifest["candidates"].setdefault(
            name,
            {
                "complete": False,
                "alpha_milli": variant[0],
                "degree": variant[1],
                "iterations": variant[2],
                "stages": {},
            },
        )
        stages = []
        for row in rows:
            key = f"{int(row['start_frame']):04d}_{int(row['end_frame']):04d}"
            source_stage_sha = stage_source_sha(row)
            old = candidate["stages"].get(key, {})
            stage = (
                old
                if valid_stage(old, source_stage_sha, binary_sha)
                else run_stage(binary, binary_sha, candidate_dir, row, variant)
            )
            candidate["stages"][key] = stage
            stages.append(stage)
            atomic_json(manifest_path, manifest)
        assembled = assemble_candidate(candidate_dir, stages)
        if assembled["decoded"]["sha256"] != source_sha or assembled["decoded"]["bytes"] != source_bytes:
            raise RuntimeError(f"assembled n600 decode differs for {name}")
        encode_seconds = sum(float(stage["encode"]["encode_seconds"]) for stage in stages)
        decode_seconds = sum(float(stage["decode"]["decode_seconds"]) for stage in stages)
        candidate.update(
            {
                "complete": True,
                "packet": assembled["packet"],
                "decoded": assembled["decoded"],
                "delta_vs_incumbent_bytes": assembled["packet"]["bytes"] - TOKEN_INCUMBENT_BYTES,
                "native_encode_seconds_sum": encode_seconds,
                "native_decode_seconds_sum": decode_seconds,
                "fallback_groups": sum(int(stage["encode"]["fallback_groups"]) for stage in stages),
                "groups": sum(int(stage["encode"]["groups"]) for stage in stages),
                "miss_symbols": sum(int(stage["encode"]["miss_symbols"]) for stage in stages),
            }
        )
        atomic_json(manifest_path, manifest)
    best_name, best = min(manifest["candidates"].items(), key=lambda item: item[1]["packet"]["bytes"])
    manifest["best_candidate"] = best_name
    manifest["best_packet_bytes"] = best["packet"]["bytes"]
    manifest["best_delta_vs_incumbent_bytes"] = best["delta_vs_incumbent_bytes"]
    manifest["complete"] = True
    manifest["completed_at_unix_s"] = time.time()
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        type=parse_variant,
        default=None,
        help="ALPHA_MILLI:DEGREE:ITERATIONS; repeat for a race",
    )
    args = parser.parse_args()
    if args.variants is None:
        args.variants = [
            (4_000, 4, 30),
            (5_000, 4, 30),
            (6_000, 4, 30),
            (8_000, 4, 30),
            (5_000, 3, 30),
            (8_000, 3, 30),
        ]
    return args


def main() -> None:
    manifest = run(parse_args())
    summary = {
        name: {
            "bytes": row["packet"]["bytes"],
            "delta_vs_incumbent_bytes": row["delta_vs_incumbent_bytes"],
            "native_decode_seconds_sum": row["native_decode_seconds_sum"],
            "fallback_groups": row["fallback_groups"],
        }
        for name, row in manifest["candidates"].items()
    }
    print(json.dumps({"best": manifest["best_candidate"], "candidates": summary}, indent=2))


if __name__ == "__main__":
    main()
