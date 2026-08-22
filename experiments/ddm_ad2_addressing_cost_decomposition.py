#!/usr/bin/env python3
"""Retained scorer-free addressing-cost decomposition for AD2.

This measurement reads the exact RC1-K2048, NR1-K32, and DX2 payloads.  It
prices assignment/address streams against empirical entropy bounds, runs small
lossless representation races, and emits ambiguity witnesses showing which
video-derived choices cannot be inferred from the other decoded surfaces.

Every byte sequence materialized by this program is retained under the output
root.  No scorer, candidate runtime, or upstream file is invoked or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import os
import shutil
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.optimization import nr1_taskcell_quotient as nr1
from tac.optimization import rc1_terminal_program_vq as rc1

DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6")
RC1_RESULT = Path("/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json")
RC1_ROOT = RC1_RESULT.parent / "retained/candidates/k2048_i3"
RC1_PAYLOAD = RC1_ROOT / "receiver/tokens.rc1v"
RC1_DECODED = RC1_ROOT / "receiver/decoded_tokens.u8"
RC1_ASSIGNMENTS = RC1_ROOT / "model/assignments.u16le"
RC1_CODEBOOK = RC1_ROOT / "model/codebook.u8"
NR1_RESULT = Path("/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1/RESULT.json")
NR1_PACKET = NR1_RESULT.parent / "retained/coder/nr1_packet.bin"
DX2_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/archive.zip")
DC1_RESULT = Path("/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep/retained/result.json")
NI1_RESULT = Path("/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/RESULT.json")
NI1_ARCHIVE = NI1_RESULT.parent / "runtime/archive.zip"

EXPECTED = {
    "rc1_payload": "eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164",
    "rc1_result": "d51e92a37bddca462a381eec66f4dbc37ff4a38f941fe2e033fc51c3c31e119c",
    "nr1_packet": "a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28",
    "nr1_result": "d3e7d58c286c82813d0356f3681b76e940d3bb206e88eaad8feebe0c68ace623",
    "rc1_module": "6c2ea6f324ea32b21d8cc079bb327c6af97e283cc963ec610859f1f2b0cbfbc9",
    "nr1_module": "66500b813eeafeaf264d57ecb47ef68360956ec1bdb040043456f3d6f101cbb6",
    "dx2_archive": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    "ni1_result": "44b41b81fd0c95ae2a2b4619abece7fdc5007d1543f5f64996456bc1c6daf938",
    "ni1_archive": "fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e",
}

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20}]
RATE_PER_BYTE = 25.0 / 37_545_489


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class Retainer:
    """Atomically retain every materialized payload and record its custody."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts: list[dict[str, Any]] = []

    def bytes(self, relative: str, payload: bytes | memoryview) -> dict[str, Any]:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        materialized = bytes(payload)
        if target.exists():
            if target.read_bytes() != materialized:
                raise RuntimeError(f"refusing to overwrite differing retained payload: {target}")
        else:
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                handle.write(materialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        receipt = file_receipt(target)
        self.artifacts.append(receipt)
        return receipt

    def json(self, relative: str, value: Any) -> dict[str, Any]:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        return self.bytes(relative, payload)

    def existing(self, relative: str) -> dict[str, Any]:
        target = self.root / relative
        if not target.is_file():
            raise RuntimeError(f"required retained artifact is absent: {target}")
        receipt = file_receipt(target)
        self.artifacts.append(receipt)
        return receipt

    def stream_frames(self, relative: str, frames: Any) -> dict[str, Any]:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            receipt = file_receipt(target)
            digest = hashlib.sha256()
            total = 0
            for frame in frames:
                payload = memoryview(np.ascontiguousarray(frame))
                digest.update(payload)
                total += payload.nbytes
            if total != receipt["bytes"] or digest.hexdigest() != receipt["sha256"]:
                raise RuntimeError(f"retained frame stream differs on resume: {target}")
            self.artifacts.append(receipt)
            return receipt
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            for frame in frames:
                handle.write(memoryview(np.ascontiguousarray(frame)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        receipt = file_receipt(target)
        self.artifacts.append(receipt)
        return receipt


def assert_pin(label: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    receipt = file_receipt(path)
    if receipt["sha256"] != expected_sha256:
        raise RuntimeError(f"inherited-state drift for {label}: {receipt['sha256']} != {expected_sha256}")
    return receipt


def entropy_bits(counts: np.ndarray) -> float:
    positive = np.asarray(counts, dtype=np.float64)
    positive = positive[positive > 0]
    total = float(positive.sum())
    if total == 0:
        return 0.0
    return float(np.sum(positive * (math.log2(total) - np.log2(positive))))


def retain_entropy(
    retainer: Retainer,
    name: str,
    symbols: np.ndarray,
    alphabet: int,
    contexts: np.ndarray,
    context_count: int,
    context_rule: str,
    coded_bytes: int | None,
) -> dict[str, Any]:
    flat_symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    flat_contexts = np.asarray(contexts, dtype=np.int64).reshape(-1)
    if flat_symbols.shape != flat_contexts.shape or flat_symbols.size == 0:
        raise ValueError(f"invalid entropy arrays for {name}")
    if int(flat_symbols.min()) < 0 or int(flat_symbols.max()) >= alphabet:
        raise ValueError(f"symbol outside alphabet for {name}")
    if int(flat_contexts.min()) < 0 or int(flat_contexts.max()) >= context_count:
        raise ValueError(f"context outside alphabet for {name}")
    memoryless = np.bincount(flat_symbols, minlength=alphabet).astype(np.int64)
    joint = np.bincount(
        flat_contexts * alphabet + flat_symbols,
        minlength=context_count * alphabet,
    ).reshape(context_count, alphabet)
    conditional_bits = float(sum(entropy_bits(row) for row in joint))
    memoryless_bits = entropy_bits(memoryless)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        memoryless_counts=memoryless,
        context_symbol_counts=joint,
    )
    counts_receipt = retainer.bytes(f"retained/entropy/{name}_counts.npz", buffer.getvalue())
    report: dict[str, Any] = {
        "symbols": int(flat_symbols.size),
        "alphabet": alphabet,
        "context_count": context_count,
        "context_rule": context_rule,
        "memoryless_plugin_bits": memoryless_bits,
        "memoryless_plugin_ceiling_bytes": math.ceil(memoryless_bits / 8.0),
        "context_model_plugin_bound_bits": conditional_bits,
        "context_model_plugin_bound_ceiling_bytes": math.ceil(conditional_bits / 8.0),
        "counts": counts_receipt,
        "bound_scope": (
            "ideal empirical data term for the declared first-order context model only; probability-"
            "model description and finite-sample redundancy are excluded, and a richer-context "
            "coder can beat this reference, so it is neither a universal lower bound nor a "
            "shippable size"
        ),
    }
    if coded_bytes is not None:
        conditional_ceiling = report["context_model_plugin_bound_ceiling_bytes"]
        report.update(
            {
                "incumbent_coded_bytes": coded_bytes,
                "incumbent_gap_to_context_model_bound_bytes": coded_bytes - conditional_ceiling,
                "incumbent_gap_to_context_model_bound_fraction": ((coded_bytes - conditional_ceiling) / coded_bytes),
            }
        )
    retainer.json(f"retained/entropy/{name}_report.json", report)
    return report


def previous_context(values: np.ndarray, alphabet: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    contexts = np.empty_like(flat)
    contexts[0] = alphabet
    contexts[1:] = flat[:-1]
    return contexts


def row_left_context(values: np.ndarray, alphabet: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("row-left context requires a matrix")
    contexts = np.empty_like(array)
    contexts[:, 0] = alphabet
    contexts[:, 1:] = array[:, :-1]
    return contexts


def temporal_context(values: np.ndarray, alphabet: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 3:
        raise ValueError("temporal context requires (time,y,x)")
    contexts = np.empty_like(array)
    contexts[0].fill(alphabet)
    contexts[1:] = array[:-1]
    return contexts


def compress_payload(raw: bytes, coder: str) -> bytes:
    if coder == "raw":
        return raw
    if coder == "brotli_q11":
        return bytes(brotli.compress(raw, quality=11))
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    if coder == "lzma1_1m":
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise ValueError(coder)


def decompress_payload(coded: bytes, coder: str) -> bytes:
    if coder == "raw":
        return coded
    if coder == "brotli_q11":
        return bytes(brotli.decompress(coded))
    if coder == "zlib9":
        return zlib.decompress(coded)
    if coder == "lzma1_1m":
        return lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise ValueError(coder)


def retain_coder_race(
    retainer: Retainer,
    family: str,
    form: str,
    raw: bytes,
) -> dict[str, Any]:
    raw_receipt = retainer.bytes(f"retained/representation/{family}/{form}/raw.bin", raw)
    variants: list[dict[str, Any]] = []
    for coder in ("raw", "zlib9", "lzma1_1m", "brotli_q11"):
        coded = compress_payload(raw, coder)
        repeat = compress_payload(raw, coder)
        if coded != repeat or decompress_payload(coded, coder) != raw:
            raise RuntimeError(f"non-deterministic or non-roundtripping {family}/{form}/{coder}")
        receipt = retainer.bytes(f"retained/representation/{family}/{form}/{coder}.bin", coded)
        repeat_receipt = retainer.bytes(f"retained/representation/{family}/{form}/{coder}.repeat.bin", repeat)
        variants.append(
            {
                "coder": coder,
                "bytes": len(coded),
                "payload": receipt,
                "repeat": repeat_receipt,
            }
        )
    winner = min(variants, key=lambda item: (item["bytes"], item["coder"]))
    return {"form": form, "raw": raw_receipt, "variants": variants, "winner": winner}


def bitpack(values: np.ndarray, width: int) -> bytes:
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size and int(flat.max()) >= 1 << width:
        raise ValueError("bitpack width is too small")
    shifts = np.arange(width, dtype=np.uint64)
    bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="little").tobytes()


def bitunpack(payload: bytes, count: int, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    bits = bits[: count * width].reshape(count, width).astype(np.uint64)
    weights = (1 << np.arange(width, dtype=np.uint64))[None, :]
    return np.sum(bits * weights, axis=1)


def block8_order(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    h, w = values.shape[-2:]
    if h % 8 or w % 8:
        raise ValueError("block8 requires dimensions divisible by eight")
    prefix = values.shape[:-2]
    return values.reshape(*prefix, h // 8, 8, w // 8, 8).swapaxes(-3, -2).reshape(*prefix, h * w)


def unblock8_order(values: np.ndarray, prefix: tuple[int, ...], h: int, w: int) -> np.ndarray:
    return np.asarray(values).reshape(*prefix, h // 8, w // 8, 8, 8).swapaxes(-3, -2).reshape(*prefix, h, w)


def parse_events(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magic, count = struct.unpack_from(">4sI", raw)
    if magic != b"QEV1" or count <= 0:
        raise ValueError("invalid QEVENT")
    cursor = 8
    previous = -1
    indices: list[int] = []
    gaps: list[int] = []
    values: list[int] = []
    for _ in range(count):
        gap, cursor = nr1._decode_uleb(raw, cursor)
        index = previous + 1 + gap
        if cursor >= len(raw):
            raise ValueError("truncated QEVENT value")
        indices.append(index)
        gaps.append(gap)
        values.append(raw[cursor])
        cursor += 1
        previous = index
    if cursor != len(raw):
        raise ValueError("QEVENT trailing bytes")
    return (
        np.asarray(indices, dtype=np.uint32),
        np.asarray(gaps, dtype=np.uint32),
        np.asarray(values, dtype=np.uint8),
    )


def split_event_form(gaps: np.ndarray, values: np.ndarray) -> bytes:
    addresses = b"".join(nr1._encode_uleb(int(gap)) for gap in gaps)
    return struct.pack(">4sII", b"QES1", len(values), len(addresses)) + addresses + values.tobytes()


def verify_split_event_form(raw: bytes, indices: np.ndarray, values: np.ndarray) -> None:
    magic, count, address_bytes = struct.unpack_from(">4sII", raw)
    if magic != b"QES1" or count != len(values):
        raise ValueError("invalid split QEVENT header")
    cursor = 12
    end = cursor + address_bytes
    gaps = []
    while cursor < end:
        gap, cursor = nr1._decode_uleb(raw, cursor)
        gaps.append(gap)
    rebuilt = np.cumsum(np.asarray(gaps, dtype=np.int64) + 1) - 1
    if not np.array_equal(rebuilt.astype(np.uint32), indices):
        raise ValueError("split QEVENT addresses do not invert")
    if raw[end:] != values.tobytes():
        raise ValueError("split QEVENT values do not invert")


def retain_nr1_changed_packet(
    retainer: Retainer,
    label: str,
    parsed: nr1.ParsedPacket,
    changed_section: nr1.Section,
    changed_raw: bytes,
) -> tuple[bytes, dict[str, Any]]:
    raw_sections = {section.name: section.raw for section in parsed.sections}
    raw_sections[changed_section] = changed_raw
    selected = {section.name: nr1.CodedCandidate(section.coder, section.coded) for section in parsed.sections}
    candidates = nr1.coder_candidates(changed_raw)
    candidate_rows = []
    for candidate in candidates:
        payload_receipt = retainer.bytes(
            f"retained/ambiguity/{label}/coder_{int(candidate.coder)}.bin",
            candidate.payload,
        )
        repeat_candidates = nr1.coder_candidates(changed_raw)
        repeat = next(item for item in repeat_candidates if item.coder == candidate.coder)
        repeat_receipt = retainer.bytes(
            f"retained/ambiguity/{label}/coder_{int(candidate.coder)}.repeat.bin",
            repeat.payload,
        )
        if repeat.payload != candidate.payload:
            raise RuntimeError("NR1 coder repeat differs")
        candidate_rows.append(
            {
                "coder": int(candidate.coder),
                "bytes": len(candidate.payload),
                "payload": payload_receipt,
                "repeat": repeat_receipt,
            }
        )
    chosen = nr1.choose_candidate(candidates)
    selected[changed_section] = chosen
    packet = nr1.build_packet(
        raw_sections,
        parsed.pair_count,
        parsed.height,
        parsed.width,
        selected=selected,
    )
    packet_receipt = retainer.bytes(f"retained/ambiguity/{label}/packet.bin", packet)
    repeat_receipt = retainer.bytes(f"retained/ambiguity/{label}/packet.repeat.bin", packet)
    return packet, {
        "changed_section": changed_section.value,
        "coder_race": candidate_rows,
        "packet": packet_receipt,
        "repeat": repeat_receipt,
    }


def write_nr1_decoded(
    retainer: Retainer,
    label: str,
    packet: bytes,
    base_path: Path | None,
) -> tuple[dict[str, Any], int | None]:
    decoded = nr1.decode_packet(packet)
    decoded.trace.require_exact_once()
    receipt = retainer.bytes(
        f"retained/ambiguity/{label}/decoded_tokens.u8",
        memoryview(np.ascontiguousarray(decoded.tokens)),
    )
    differences = None
    if base_path is not None:
        base = np.memmap(base_path, dtype=np.uint8, mode="r", shape=decoded.tokens.shape)
        differences = int(np.count_nonzero(decoded.tokens != base))
        del base
    return receipt, differences


def cold_store_superseded_measurements(
    measurement_parent: Path,
    keep_root: Path,
    cold_store_root: Path,
) -> dict[str, Any]:
    """Copy-verify-delete complete superseded AD2 trees, then leave symlinks."""
    from comma_lab.artifact_retention import _copy_verify_then_delete, directory_digest

    parent = measurement_parent.resolve()
    keep = keep_root.resolve()
    cold = cold_store_root.resolve()
    if parent != DEFAULT_OUTPUT.parent.resolve() or keep.parent != parent or keep.name != "measurement_v6":
        raise RuntimeError("cold-store cleanup is pinned to AD2 measurement_v6 siblings")
    cold.mkdir(parents=True, exist_ok=True)
    execution_path = parent / "COLD_STORE_EXECUTION.json"
    if execution_path.is_file():
        existing = json.loads(execution_path.read_text())
        if (
            existing.get("schema") != "ddm.ad2.cold_store_execution.v1"
            or existing.get("authoritative_keep_root") != str(keep)
            or existing.get("cold_store_root") != str(cold)
        ):
            raise RuntimeError("retained cold-store execution identity differs")
        for row in existing["rows"]:
            source = Path(row["source"])
            destination = Path(row["destination"])
            if (
                not source.is_symlink()
                or source.resolve() != destination
                or directory_digest(destination)["sha256"] != row["source_digest"]["sha256"]
            ):
                raise RuntimeError(f"retained cold-store execution drifted: {source}")
        return existing
    if shutil.disk_usage(cold).free < 8 * (1 << 30):
        raise RuntimeError("cold-store cleanup requires at least 8 GiB free on destination tier")

    selected = [parent / f"measurement_v{version}" for version in range(2, 6)]
    plan_rows: list[dict[str, Any]] = []
    for source in selected:
        destination = cold / source.name
        if source.is_symlink():
            source_digest = directory_digest(destination)
            plan_rows.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "status": "certified_for_copy_verify_delete",
                    "source_digest": source_digest,
                    "rebuild_command": (f"{sys.executable} {Path(__file__).resolve()} --output-root {source}"),
                    "reason": "complete superseded deterministic AD2 measurement; v6 is authoritative",
                }
            )
            continue
        result_path = source / "RESULT.json"
        if not result_path.is_file():
            raise RuntimeError(f"superseded tree has no completed result: {source}")
        result = json.loads(result_path.read_text())
        if result.get("schema") != "ddm.ad2.addressing_cost_decomposition.v1":
            raise RuntimeError(f"superseded result schema differs: {result_path}")
        for artifact in result["retention"]["artifacts"]:
            artifact_path = Path(artifact["path"])
            if file_receipt(artifact_path) != artifact:
                raise RuntimeError(f"superseded artifact drifted: {artifact_path}")
        digest = directory_digest(source)
        plan_rows.append(
            {
                "source": str(source),
                "destination": str(destination),
                "status": "certified_for_copy_verify_delete",
                "source_digest": digest,
                "rebuild_command": (f"{sys.executable} {Path(__file__).resolve()} --output-root {source}"),
                "reason": "complete superseded deterministic AD2 measurement; v6 is authoritative",
            }
        )
    manifest_root = Retainer(parent)
    plan = {
        "schema": "ddm.ad2.cold_store_plan.v1",
        "authoritative_keep_root": str(keep),
        "cold_store_root": str(cold),
        "rows": plan_rows,
        "score_claim": False,
        "policy": "copy, verify full tree digest, then delete source and leave a source-path symlink",
    }
    manifest_root.json("COLD_STORE_PLAN.json", plan)

    execution_rows: list[dict[str, Any]] = []
    for row in plan_rows:
        source = Path(row["source"])
        destination = Path(row["destination"])
        if source.is_symlink():
            if (
                source.resolve() != destination
                or directory_digest(destination)["sha256"] != row["source_digest"]["sha256"]
            ):
                raise RuntimeError(f"preexisting cold-store link differs: {source}")
            execution_rows.append(
                {
                    **row,
                    "status": "moved_and_source_symlinked",
                    "verification": {"method": "preexisting_verified_symlink"},
                    "source_symlink_target": str(source.readlink()),
                }
            )
            continue
        verification = _copy_verify_then_delete(
            source,
            destination,
            repo_root=REPO,
            bytes_estimate=int(row["source_digest"]["bytes"]),
            allowed_source_roots=(parent,),
        )
        source.symlink_to(destination, target_is_directory=True)
        if directory_digest(destination)["sha256"] != row["source_digest"]["sha256"]:
            raise RuntimeError(f"cold-store final digest differs: {destination}")
        execution_rows.append(
            {
                **row,
                "status": "moved_and_source_symlinked",
                "verification": verification,
                "source_symlink_target": str(source.readlink()),
            }
        )
    execution = {
        "schema": "ddm.ad2.cold_store_execution.v1",
        "authoritative_keep_root": str(keep),
        "cold_store_root": str(cold),
        "rows": execution_rows,
        "score_claim": False,
        "local_bytes_reclaimed": sum(
            int(row.get("source_digest", {}).get("bytes", 0))
            for row in execution_rows
            if row["status"] == "moved_and_source_symlinked"
        ),
    }
    manifest_root.json("COLD_STORE_EXECUTION.json", execution)
    return execution


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retainer = Retainer(output)
    free = shutil.disk_usage(output).free
    if free < 9 * (1 << 30):
        raise RuntimeError(f"storage preflight failed: only {free} bytes free")
    preflight_path = output / "STORAGE_PREFLIGHT.json"
    if preflight_path.exists():
        storage = json.loads(preflight_path.read_text())
        if storage.get("path") != str(output) or storage.get("minimum_free_bytes") != 9 * (1 << 30):
            raise RuntimeError("retained storage preflight differs on resume")
        retainer.existing("STORAGE_PREFLIGHT.json")
    else:
        storage = {
            "path": str(output),
            "free_bytes_at_launch": free,
            "minimum_free_bytes": 9 * (1 << 30),
            "estimated_new_bytes": 1 * (1 << 30),
            "policy": "APDataStore SSD; fail closed; no automatic deletion",
        }
        retainer.json("STORAGE_PREFLIGHT.json", storage)

    pins = {
        "rc1_result": assert_pin("RC1 result", RC1_RESULT, EXPECTED["rc1_result"]),
        "rc1_payload": assert_pin("RC1 payload", RC1_PAYLOAD, EXPECTED["rc1_payload"]),
        "nr1_result": assert_pin("NR1 result", NR1_RESULT, EXPECTED["nr1_result"]),
        "nr1_packet": assert_pin("NR1 packet", NR1_PACKET, EXPECTED["nr1_packet"]),
        "rc1_module": assert_pin(
            "RC1 module",
            REPO / "src/tac/optimization/rc1_terminal_program_vq.py",
            EXPECTED["rc1_module"],
        ),
        "nr1_module": assert_pin(
            "NR1 module",
            REPO / "src/tac/optimization/nr1_taskcell_quotient.py",
            EXPECTED["nr1_module"],
        ),
        "dx2_archive": assert_pin("DX2 archive", DX2_ARCHIVE, EXPECTED["dx2_archive"]),
        "ni1_result": assert_pin("NI1 result", NI1_RESULT, EXPECTED["ni1_result"]),
        "ni1_archive": assert_pin("NI1 archive", NI1_ARCHIVE, EXPECTED["ni1_archive"]),
        "dc1_result": file_receipt(DC1_RESULT),
        "producer_source": file_receipt(Path(__file__).resolve()),
    }
    retainer.json("stage_01_inherited_pins.json", pins)
    retainer.bytes("retained/source/producer.py", Path(__file__).read_bytes())

    dc1 = json.loads(DC1_RESULT.read_text())
    dc1_position_bits = sum(int(group["chosen"]["position_bits"]) for group in dc1["groups"])
    dc1_length_bits = sum(int(group["chosen"]["hash_length_bits"]) for group in dc1["groups"])

    rc1_payload = RC1_PAYLOAD.read_bytes()
    (
        _magic,
        _version,
        rc1_assignment_method,
        rc1_codebook_method,
        _flags,
        _height,
        _width,
        _time_steps,
        rc1_k,
        rc1_assignment_bytes,
        rc1_codebook_bytes,
        _assignment_crc,
        _codebook_crc,
        rc1_decoded_digest,
    ) = rc1.PAYLOAD_HEADER.unpack_from(rc1_payload)
    rc1_model = rc1.TokenVQModel(
        np.fromfile(RC1_ASSIGNMENTS, dtype="<u2").reshape(_height, _width),
        np.fromfile(RC1_CODEBOOK, dtype=np.uint8).reshape(rc1_k, _time_steps),
    )
    rc1_model.validate()
    canonical_rc1 = rc1.canonicalize_model(rc1_model)
    if not (
        np.array_equal(canonical_rc1.assignments, rc1_model.assignments)
        and np.array_equal(canonical_rc1.codebook, rc1_model.codebook)
    ):
        raise RuntimeError("retained RC1 model is not canonical")
    rc1_decoded_sha = rc1_decoded_digest.hex()
    if sha256_file(RC1_DECODED) != rc1_decoded_sha:
        raise RuntimeError("retained RC1 decoded bytes differ from payload digest")
    rc1_assignment_coded = rc1_payload[rc1.PAYLOAD_HEADER.size : rc1.PAYLOAD_HEADER.size + rc1_assignment_bytes]
    rc1_codebook_coded = rc1_payload[rc1.PAYLOAD_HEADER.size + rc1_assignment_bytes :]
    retainer.bytes("retained/input/rc1_payload.rc1v", rc1_payload)
    retainer.bytes("retained/input/rc1_assignment_coded.bin", rc1_assignment_coded)
    retainer.bytes("retained/input/rc1_codebook_coded.bin", rc1_codebook_coded)
    retainer.bytes(
        "retained/input/rc1_assignments.u16le",
        np.asarray(rc1_model.assignments, dtype="<u2").tobytes(),
    )
    retainer.bytes("retained/input/rc1_codebook.u8", rc1_model.codebook.tobytes())

    nr1_packet = NR1_PACKET.read_bytes()
    parsed_nr1 = nr1.parse_packet(nr1_packet)
    nr1_sections = {section.name: section for section in parsed_nr1.sections}
    retainer.bytes("retained/input/nr1_packet.bin", nr1_packet)
    for section in parsed_nr1.sections:
        retainer.bytes(f"retained/input/nr1_{section.name.value}_raw.bin", section.raw)
        retainer.bytes(f"retained/input/nr1_{section.name.value}_coded.bin", section.coded)

    with zipfile.ZipFile(DX2_ARCHIVE) as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("DX2 archive member list drifted")
        dx2_member = archive.read("p")
    header = rc1.RX1_MODEL_HEADER.unpack_from(dx2_member)
    magic, version, dx2_codec, table_mode, reserved, hpac_len, semantic_len, carrier_len = header
    if magic != rc1.RX1_MAGIC or version != 1:
        raise RuntimeError("DX2 RX1 header drifted")
    cursor = rc1.RX1_MODEL_HEADER.size
    dx2_hpac = dx2_member[cursor : cursor + hpac_len]
    cursor += hpac_len
    dx2_semantic = dx2_member[cursor : cursor + semantic_len]
    cursor += semantic_len
    dx2_carrier = dx2_member[cursor : cursor + carrier_len]
    cursor += carrier_len
    dx2_residual = dx2_member[cursor : cursor + rc1.DX2_RESIDUAL_STREAM_BYTES]
    cursor += rc1.DX2_RESIDUAL_STREAM_BYTES
    dx2_tokens = dx2_member[cursor:]
    dx2_zip_framing = DX2_ARCHIVE.stat().st_size - len(dx2_member)
    for name, payload in (
        ("member", dx2_member),
        ("hpac", dx2_hpac),
        ("semantic", dx2_semantic),
        ("carrier", dx2_carrier),
        ("residual", dx2_residual),
        ("tokens", dx2_tokens),
    ):
        retainer.bytes(f"retained/input/dx2_{name}.bin", payload)
    if dx2_codec != 2:
        raise RuntimeError(f"DX2 codec {dx2_codec} is not the pinned Brotli codec")
    dx2_hpac_raw = bytes(brotli.decompress(dx2_hpac))
    retainer.bytes("retained/input/dx2_hpac_raw.bin", dx2_hpac_raw)

    nr1_qctx_raw = nr1_sections[nr1.Section.QCTX].raw
    _, gh, gw = struct.unpack_from(">4sHH", nr1_qctx_raw)
    qctx = np.frombuffer(nr1_qctx_raw, dtype=np.uint8, offset=8).reshape(gh, gw)
    nr1_qpair_raw = nr1_sections[nr1.Section.QPAIR].raw
    _, pair_count, pair_gh, pair_gw = struct.unpack_from(">4sHHH", nr1_qpair_raw)
    qpair = np.frombuffer(nr1_qpair_raw, dtype=np.uint8, offset=10).reshape(pair_count, pair_gh, pair_gw)
    qevent_indices, qevent_gaps, qevent_values = parse_events(nr1_sections[nr1.Section.QEVENT].raw)

    entropy = {
        "rc1_assignment": retain_entropy(
            retainer,
            "rc1_assignment",
            rc1_model.assignments,
            int(rc1_k),
            row_left_context(rc1_model.assignments, int(rc1_k)),
            int(rc1_k) + 1,
            "causal left assignment ID within each raster row; row-start sentinel",
            int(rc1_assignment_bytes),
        ),
        "nr1_qctx": retain_entropy(
            retainer,
            "nr1_qctx",
            qctx,
            32,
            row_left_context(qctx, 32),
            33,
            "causal left baseline ID within each tile-grid raster row; row-start sentinel",
            len(nr1_sections[nr1.Section.QCTX].coded),
        ),
        "nr1_qpair": retain_entropy(
            retainer,
            "nr1_qpair",
            qpair,
            34,
            temporal_context(qpair, 34),
            35,
            "same-tile QPAIR symbol at the previous pair; pair-zero sentinel",
            len(nr1_sections[nr1.Section.QPAIR].coded),
        ),
        "dx2_hpac_raw_bytes": retain_entropy(
            retainer,
            "dx2_hpac_raw_bytes",
            np.frombuffer(dx2_hpac_raw, dtype=np.uint8),
            256,
            previous_context(np.frombuffer(dx2_hpac_raw, dtype=np.uint8), 256),
            257,
            "immediately preceding raw HPAC byte; stream-start sentinel",
            len(dx2_hpac),
        ),
    }
    qevent_gap_counts = Counter(map(int, qevent_gaps))
    qevent_context_counts: dict[int, Counter[int]] = defaultdict(Counter)
    previous_length = 0
    for gap in map(int, qevent_gaps):
        qevent_context_counts[previous_length][gap] += 1
        previous_length = len(nr1._encode_uleb(gap))
    qevent_memoryless_bits = entropy_bits(np.asarray(list(qevent_gap_counts.values())))
    qevent_conditional_bits = sum(
        entropy_bits(np.asarray(list(counts.values()))) for counts in qevent_context_counts.values()
    )
    qevent_entropy = {
        "address_symbols": len(qevent_gaps),
        "memoryless_plugin_bits": qevent_memoryless_bits,
        "memoryless_plugin_ceiling_bytes": math.ceil(qevent_memoryless_bits / 8),
        "context_model_plugin_bound_bits": qevent_conditional_bits,
        "context_model_plugin_bound_ceiling_bytes": math.ceil(qevent_conditional_bits / 8),
        "context_rule": "previous coordinate-gap ULEB byte length, sentinel zero initially",
        "coded_bytes": None,
        "why_no_coded_comparison": (
            "QEVENT interleaves gap/address bytes and class-value payload in one Brotli stream; "
            "coded attribution is not identifiable"
        ),
        "bound_scope": (
            "ideal empirical address data term for the declared context model; probability-model "
            "description excluded; not a universal or shippable lower bound"
        ),
    }
    retainer.json(
        "retained/entropy/nr1_qevent_gap_counts.json",
        {
            "memoryless": {str(key): value for key, value in sorted(qevent_gap_counts.items())},
            "conditioned": {
                str(context): {str(key): value for key, value in sorted(counts.items())}
                for context, counts in sorted(qevent_context_counts.items())
            },
        },
    )
    retainer.json("retained/entropy/nr1_qevent_report.json", qevent_entropy)
    entropy["nr1_qevent_addresses_only"] = qevent_entropy

    trials: dict[str, Any] = {}
    rc1_forms: list[tuple[str, bytes]] = []
    rc1_block = block8_order(rc1_model.assignments).reshape(-1)
    if not np.array_equal(unblock8_order(rc1_block, (), *rc1_model.assignments.shape), rc1_model.assignments):
        raise RuntimeError("RC1 block8 form does not invert")
    rc1_forms.append(("block8_u16le", rc1_block.astype("<u2").tobytes()))
    rc1_packed = bitpack(rc1_model.assignments, 11)
    if not np.array_equal(
        bitunpack(rc1_packed, rc1_model.assignments.size, 11).reshape(rc1_model.assignments.shape),
        rc1_model.assignments,
    ):
        raise RuntimeError("RC1 11-bit form does not invert")
    rc1_forms.append(("raster_fixed11", rc1_packed))
    rc1_block_packed = bitpack(rc1_block, 11)
    if not np.array_equal(bitunpack(rc1_block_packed, rc1_block.size, 11), rc1_block):
        raise RuntimeError("RC1 block8 fixed11 form does not invert")
    rc1_forms.append(("block8_fixed11", rc1_block_packed))
    rc1_trial_rows = [retain_coder_race(retainer, "rc1_assignment", name, raw) for name, raw in rc1_forms]
    rc1_best = min(
        [{"form": "incumbent_row_u16", "bytes": int(rc1_assignment_bytes), "coder": "brotli_q11"}]
        + [
            {"form": row["form"], "bytes": row["winner"]["bytes"], "coder": row["winner"]["coder"]}
            for row in rc1_trial_rows
        ],
        key=lambda row: (row["bytes"], row["form"]),
    )
    trials["rc1_assignment"] = {
        "incumbent_coded_bytes": int(rc1_assignment_bytes),
        "forms": rc1_trial_rows,
        "winner": rc1_best,
        "measured_substream_delta_bytes": int(rc1_assignment_bytes) - rc1_best["bytes"],
        "integration_status": "not integrated; live RC1 payload untouched",
    }

    qctx_forms: list[tuple[str, bytes]] = []
    qctx_block = block8_order(qctx).reshape(-1)
    if not np.array_equal(unblock8_order(qctx_block, (), gh, gw), qctx):
        raise RuntimeError("QCTX block8 form does not invert")
    qctx_forms.append(("block8_u8", nr1_qctx_raw[:8] + qctx_block.tobytes()))
    qctx_packed = bitpack(qctx, 5)
    if not np.array_equal(bitunpack(qctx_packed, qctx.size, 5).reshape(qctx.shape), qctx):
        raise RuntimeError("QCTX fixed5 form does not invert")
    qctx_forms.append(("raster_fixed5", nr1_qctx_raw[:8] + qctx_packed))
    qctx_block_packed = bitpack(qctx_block, 5)
    if not np.array_equal(bitunpack(qctx_block_packed, qctx_block.size, 5), qctx_block):
        raise RuntimeError("QCTX block8 fixed5 form does not invert")
    qctx_forms.append(("block8_fixed5", nr1_qctx_raw[:8] + qctx_block_packed))
    qctx_trial_rows = [retain_coder_race(retainer, "nr1_qctx", name, raw) for name, raw in qctx_forms]
    qctx_incumbent = len(nr1_sections[nr1.Section.QCTX].coded)
    qctx_best = min(
        [{"form": "incumbent_raster_u8", "bytes": qctx_incumbent, "coder": "brotli_q11"}]
        + [
            {"form": row["form"], "bytes": row["winner"]["bytes"], "coder": row["winner"]["coder"]}
            for row in qctx_trial_rows
        ],
        key=lambda row: (row["bytes"], row["form"]),
    )
    trials["nr1_qctx"] = {
        "incumbent_coded_bytes": qctx_incumbent,
        "forms": qctx_trial_rows,
        "winner": qctx_best,
        "measured_substream_delta_bytes": qctx_incumbent - qctx_best["bytes"],
        "integration_status": "not integrated; live NR1 packet untouched",
    }

    qpair_forms: list[tuple[str, bytes]] = []
    tile_time = qpair.transpose(1, 2, 0).reshape(-1)
    if not np.array_equal(tile_time.reshape(gh, gw, pair_count).transpose(2, 0, 1), qpair):
        raise RuntimeError("QPAIR tile-time form does not invert")
    qpair_forms.append(("tile_time_u8", nr1_qpair_raw[:10] + tile_time.tobytes()))
    qpair_packed = bitpack(qpair, 6)
    if not np.array_equal(bitunpack(qpair_packed, qpair.size, 6).reshape(qpair.shape), qpair):
        raise RuntimeError("QPAIR fixed6 form does not invert")
    qpair_forms.append(("pair_raster_fixed6", nr1_qpair_raw[:10] + qpair_packed))
    tile_time_packed = bitpack(tile_time, 6)
    if not np.array_equal(bitunpack(tile_time_packed, tile_time.size, 6), tile_time):
        raise RuntimeError("QPAIR tile-time fixed6 form does not invert")
    qpair_forms.append(("tile_time_fixed6", nr1_qpair_raw[:10] + tile_time_packed))
    qpair_block = block8_order(qpair).reshape(-1)
    if not np.array_equal(unblock8_order(qpair_block.reshape(pair_count, -1), (pair_count,), gh, gw), qpair):
        raise RuntimeError("QPAIR block8 form does not invert")
    qpair_forms.append(("pair_block8_u8", nr1_qpair_raw[:10] + qpair_block.tobytes()))
    qpair_trial_rows = [retain_coder_race(retainer, "nr1_qpair", name, raw) for name, raw in qpair_forms]
    qpair_incumbent = len(nr1_sections[nr1.Section.QPAIR].coded)
    qpair_best = min(
        [{"form": "incumbent_pair_raster_u8", "bytes": qpair_incumbent, "coder": "brotli_q11"}]
        + [
            {"form": row["form"], "bytes": row["winner"]["bytes"], "coder": row["winner"]["coder"]}
            for row in qpair_trial_rows
        ],
        key=lambda row: (row["bytes"], row["form"]),
    )
    trials["nr1_qpair"] = {
        "incumbent_coded_bytes": qpair_incumbent,
        "forms": qpair_trial_rows,
        "winner": qpair_best,
        "measured_substream_delta_bytes": qpair_incumbent - qpair_best["bytes"],
        "integration_status": "not integrated; live NR1 packet untouched",
    }

    qevent_split = split_event_form(qevent_gaps, qevent_values)
    verify_split_event_form(qevent_split, qevent_indices, qevent_values)
    qevent_trials = [retain_coder_race(retainer, "nr1_qevent", "split_addresses_values", qevent_split)]
    qevent_incumbent = len(nr1_sections[nr1.Section.QEVENT].coded)
    qevent_best = min(
        [{"form": "incumbent_interleaved", "bytes": qevent_incumbent, "coder": "brotli_q11"}]
        + [
            {"form": row["form"], "bytes": row["winner"]["bytes"], "coder": row["winner"]["coder"]}
            for row in qevent_trials
        ],
        key=lambda row: (row["bytes"], row["form"]),
    )
    trials["nr1_qevent"] = {
        "incumbent_coded_bytes": qevent_incumbent,
        "forms": qevent_trials,
        "winner": qevent_best,
        "measured_substream_delta_bytes": qevent_incumbent - qevent_best["bytes"],
        "integration_status": "not integrated; live NR1 packet untouched",
    }

    ambiguity: dict[str, Any] = {}
    rc1_alt_assignments = rc1_model.assignments.copy()
    differing = np.argwhere(rc1_alt_assignments[:, 1:] != rc1_alt_assignments[:, :-1])
    if len(differing) == 0:
        raise RuntimeError("RC1 assignment map has no distinct adjacent values")
    y, x_minus_one = map(int, differing[0])
    x = x_minus_one + 1
    rc1_alt_assignments[y, x_minus_one], rc1_alt_assignments[y, x] = (
        rc1_alt_assignments[y, x],
        rc1_alt_assignments[y, x_minus_one],
    )
    rc1_alt_model = rc1.TokenVQModel(rc1_alt_assignments, rc1_model.codebook.copy())
    rc1_assignment_variants = rc1.encode_assignment_variants(rc1_alt_model.assignments, rc1_k)
    rc1_codebook_variants = rc1.encode_codebook_variants(rc1_alt_model.codebook)
    for variant in rc1_assignment_variants:
        retainer.bytes(
            f"retained/ambiguity/rc1_assignment/coder_assignment_{variant.method_id:03d}.bin",
            variant.payload,
        )
    for variant in rc1_codebook_variants:
        retainer.bytes(
            f"retained/ambiguity/rc1_assignment/coder_codebook_{variant.method_id:03d}.bin",
            variant.payload,
        )
    assignment_selected = next(
        variant for variant in rc1_assignment_variants if variant.method_id == rc1_assignment_method
    )
    codebook_selected = next(variant for variant in rc1_codebook_variants if variant.method_id == rc1_codebook_method)
    rc1_alt_sha = rc1.decoded_sha256(rc1_alt_model)
    rc1_alt_payload = rc1.build_payload(
        rc1_alt_model,
        assignment_selected,
        codebook_selected,
        rc1_alt_sha,
    )
    rc1_alt_receipt = retainer.bytes("retained/ambiguity/rc1_assignment/payload.rc1v", rc1_alt_payload)
    rc1_alt_repeat = retainer.bytes("retained/ambiguity/rc1_assignment/payload.repeat.rc1v", rc1_alt_payload)
    parsed_alt, parsed_alt_sha = rc1.parse_payload(rc1_alt_payload)
    if parsed_alt_sha != rc1_alt_sha:
        raise RuntimeError("RC1 ambiguity payload SHA field differs")
    rc1_alt_decoded = retainer.stream_frames(
        "retained/ambiguity/rc1_assignment/decoded_tokens.u8",
        rc1.iter_decoded_frames(parsed_alt),
    )
    if rc1_alt_decoded["sha256"] != rc1_alt_sha:
        raise RuntimeError("RC1 ambiguity decoded bytes differ from declared digest")
    base_tokens = np.memmap(
        RC1_DECODED,
        dtype=np.uint8,
        mode="r",
        shape=(rc1_model.codebook.shape[1], *rc1_model.assignments.shape),
    )
    alt_tokens = np.memmap(
        Path(rc1_alt_decoded["path"]),
        dtype=np.uint8,
        mode="r",
        shape=base_tokens.shape,
    )
    rc1_diff = int(np.count_nonzero(base_tokens != alt_tokens))
    del base_tokens, alt_tokens
    if rc1_diff <= 0:
        raise RuntimeError("RC1 ambiguity witness is inert")
    ambiguity["rc1_assignment"] = {
        "same_surfaces": "same canonical codebook and shape",
        "changed_surface": "two assignment IDs swapped",
        "changed_coordinates": [[y, x_minus_one], [y, x]],
        "decoded_token_differences": rc1_diff,
        "payload": rc1_alt_receipt,
        "repeat": rc1_alt_repeat,
        "decoded": rc1_alt_decoded,
        "verdict": "assignment IDs are not derivable from the codebook or generic raster coordinates",
    }

    nr1_base_decoded, _ = write_nr1_decoded(retainer, "nr1_base", nr1_packet, None)
    base_nr1_path = Path(nr1_base_decoded["path"])

    event_pair_cells = {
        (
            int(index) // (parsed_nr1.height * parsed_nr1.width),
            (int(index) % (parsed_nr1.height * parsed_nr1.width)) // parsed_nr1.width // 8,
            (int(index) % parsed_nr1.width) // 8,
        )
        for index in qevent_indices
    }
    event_cells = {(yy, xx) for _, yy, xx in event_pair_cells}
    qctx_alt = qctx.copy()
    candidate_cells = np.asarray(
        [(yy, xx) for yy, xx in np.argwhere(np.any(qpair == 1, axis=0)) if (int(yy), int(xx)) not in event_cells],
        dtype=np.int64,
    )
    if len(candidate_cells) == 0:
        raise RuntimeError("QCTX has no consumed baseline cell")
    cy, cx = map(int, candidate_cells[0])
    qctx_alt[cy, cx] = (int(qctx_alt[cy, cx]) + 1) % 32
    qctx_alt_raw = nr1_qctx_raw[:8] + qctx_alt.tobytes()
    qctx_packet, qctx_witness = retain_nr1_changed_packet(
        retainer, "nr1_qctx", parsed_nr1, nr1.Section.QCTX, qctx_alt_raw
    )
    qctx_decoded, qctx_diff = write_nr1_decoded(retainer, "nr1_qctx", qctx_packet, base_nr1_path)
    if not qctx_diff:
        raise RuntimeError("QCTX ambiguity witness is inert")
    qctx_witness.update(
        {
            "same_surfaces": "same QPARAM, QPAIR, and QEVENT",
            "changed_surface": "one consumed QCTX baseline ID",
            "changed_cell": [cy, cx],
            "decoded_token_differences": qctx_diff,
            "decoded": qctx_decoded,
            "verdict": "baseline IDs are not derivable from the other three paid surfaces",
        }
    )
    ambiguity["nr1_qctx"] = qctx_witness

    qpair_alt = qpair.copy()
    changed = None
    for pair_index in range(pair_count - 1, 0, -1):
        for yy in range(gh):
            for xx in range(gw):
                if (pair_index, yy, xx) in event_pair_cells:
                    continue
                current = int(qpair_alt[pair_index, yy, xx])
                replacement = 2 + ((current + 1) % 32)
                if replacement != current:
                    qpair_alt[pair_index, yy, xx] = replacement
                    changed = (pair_index, yy, xx, current, replacement)
                    break
            if changed:
                break
        if changed:
            break
    if changed is None:
        raise RuntimeError("could not construct QPAIR ambiguity witness")
    qpair_alt_raw = nr1_qpair_raw[:10] + qpair_alt.tobytes()
    qpair_packet, qpair_witness = retain_nr1_changed_packet(
        retainer, "nr1_qpair", parsed_nr1, nr1.Section.QPAIR, qpair_alt_raw
    )
    qpair_decoded, qpair_diff = write_nr1_decoded(retainer, "nr1_qpair", qpair_packet, base_nr1_path)
    if not qpair_diff:
        raise RuntimeError("QPAIR ambiguity witness is inert")
    qpair_witness.update(
        {
            "same_surfaces": "same QPARAM, QCTX, and QEVENT",
            "changed_surface": "one QPAIR temporal/context choice",
            "changed_coordinate_and_symbols": list(changed),
            "decoded_token_differences": qpair_diff,
            "decoded": qpair_decoded,
            "verdict": "QPAIR choices are not derivable from the other three paid surfaces",
        }
    )
    ambiguity["nr1_qpair"] = qpair_witness

    qparam = nr1._consume_qparam(nr1_sections[nr1.Section.QPARAM].raw)
    tile_height, tile_width, codebook = qparam
    _, _, baseline = nr1._consume_qctx(nr1_sections[nr1.Section.QCTX].raw, len(codebook))
    assignments = nr1._consume_qpair(
        nr1_sections[nr1.Section.QPAIR].raw,
        pair_count,
        gh,
        gw,
        baseline,
        len(codebook),
    )
    qevent_alt_values = qevent_values.copy()
    first_index = int(qevent_indices[0])
    first_pair, frame_offset = divmod(first_index, parsed_nr1.height * parsed_nr1.width)
    first_y, first_x = divmod(frame_offset, parsed_nr1.width)
    tile_y, within_y = divmod(first_y, tile_height)
    tile_x, within_x = divmod(first_x, tile_width)
    codeword = int(assignments[first_pair, tile_y, tile_x])
    pre_event_value = int(codebook[codeword, within_y * tile_width + within_x])
    forbidden = {int(qevent_values[0]), pre_event_value}
    replacement_value = next(value for value in range(5) if value not in forbidden)
    qevent_alt_values[0] = replacement_value
    qevent_alt_raw = nr1._encode_events(qevent_indices, qevent_alt_values)
    qevent_packet, qevent_witness = retain_nr1_changed_packet(
        retainer, "nr1_qevent", parsed_nr1, nr1.Section.QEVENT, qevent_alt_raw
    )
    qevent_decoded, qevent_diff = write_nr1_decoded(retainer, "nr1_qevent", qevent_packet, base_nr1_path)
    if not qevent_diff:
        raise RuntimeError("QEVENT ambiguity witness is inert")
    qevent_witness.update(
        {
            "same_surfaces": "same QPARAM, QCTX, and QPAIR",
            "changed_surface": "one QEVENT class value at the same stored coordinate",
            "changed_index": first_index,
            "decoded_token_differences": qevent_diff,
            "decoded": qevent_decoded,
            "verdict": "QEVENT corrections are not derivable from the other three paid surfaces",
        }
    )
    ambiguity["nr1_qevent"] = qevent_witness

    rc1_total = len(rc1_payload)
    nr1_total = len(nr1_packet)
    dx2_total = DX2_ARCHIVE.stat().st_size
    nr1_header_bytes = nr1._OUTER.size + len(nr1.Section) * nr1._SECTION.size
    decompositions = {
        "dc1": {
            "denominator": "190 selected sparse-grid groups",
            "position_bits": dc1_position_bits,
            "position_bytes": dc1_position_bits / 8,
            "per_block_length_bits": dc1_length_bits,
            "per_block_length_bytes": dc1_length_bits / 8,
        },
        "rc1_k2048": {
            "denominator_bytes": rc1_total,
            "rows": [
                {
                    "stream": "header/framing",
                    "class": "addressing/how-much",
                    "bytes": rc1.PAYLOAD_HEADER.size,
                    "fraction": rc1.PAYLOAD_HEADER.size / rc1_total,
                },
                {
                    "stream": "spatial assignment IDs",
                    "class": "addressing",
                    "bytes": int(rc1_assignment_bytes),
                    "fraction": int(rc1_assignment_bytes) / rc1_total,
                },
                {
                    "stream": "temporal program codebook",
                    "class": "payload/what",
                    "bytes": int(rc1_codebook_bytes),
                    "fraction": int(rc1_codebook_bytes) / rc1_total,
                },
            ],
        },
        "nr1_k32": {
            "denominator_bytes": nr1_total,
            "rows": [
                {
                    "stream": "outer and four section headers",
                    "class": "addressing/how-much/framing",
                    "bytes": nr1_header_bytes,
                    "fraction": nr1_header_bytes / nr1_total,
                },
                {
                    "stream": "QPARAM coded",
                    "class": "payload/what",
                    "bytes": len(nr1_sections[nr1.Section.QPARAM].coded),
                    "fraction": len(nr1_sections[nr1.Section.QPARAM].coded) / nr1_total,
                },
                {
                    "stream": "QCTX coded",
                    "class": "addressing",
                    "bytes": len(nr1_sections[nr1.Section.QCTX].coded),
                    "fraction": len(nr1_sections[nr1.Section.QCTX].coded) / nr1_total,
                },
                {
                    "stream": "QPAIR coded",
                    "class": "addressing",
                    "bytes": len(nr1_sections[nr1.Section.QPAIR].coded),
                    "fraction": len(nr1_sections[nr1.Section.QPAIR].coded) / nr1_total,
                },
                {
                    "stream": "QEVENT coded",
                    "class": "mixed address+class payload; coded split unidentifiable",
                    "bytes": len(nr1_sections[nr1.Section.QEVENT].coded),
                    "fraction": len(nr1_sections[nr1.Section.QEVENT].coded) / nr1_total,
                },
            ],
            "physical_attribution_including_headers": {
                section.value: end - start for section, (start, end) in nr1.physical_attribution(nr1_packet).items()
            },
        },
        "dx2": {
            "denominator_bytes": dx2_total,
            "rows": [
                {
                    "stream": "ZIP framing",
                    "class": "framing/how-much",
                    "bytes": dx2_zip_framing,
                    "fraction": dx2_zip_framing / dx2_total,
                },
                {
                    "stream": "RX1 header",
                    "class": "addressing/how-much",
                    "bytes": rc1.RX1_MODEL_HEADER.size,
                    "fraction": rc1.RX1_MODEL_HEADER.size / dx2_total,
                },
                {
                    "stream": "HPAC probability model",
                    "class": "addressing/how-to; video-derived",
                    "bytes": len(dx2_hpac),
                    "fraction": len(dx2_hpac) / dx2_total,
                },
                {
                    "stream": "semantic renderer",
                    "class": "payload/what",
                    "bytes": len(dx2_semantic),
                    "fraction": len(dx2_semantic) / dx2_total,
                },
                {
                    "stream": "carrier",
                    "class": "mixed payload+basis/coefficient metadata; coded split unidentifiable",
                    "bytes": len(dx2_carrier),
                    "fraction": len(dx2_carrier) / dx2_total,
                },
                {
                    "stream": "residual",
                    "class": "payload/what",
                    "bytes": len(dx2_residual),
                    "fraction": len(dx2_residual) / dx2_total,
                },
                {
                    "stream": "semantic token stream",
                    "class": "payload/what; raster addresses implicit",
                    "bytes": len(dx2_tokens),
                    "fraction": len(dx2_tokens) / dx2_total,
                },
            ],
            "header_fields": {
                "codec": dx2_codec,
                "table_mode": table_mode,
                "feature_bits": reserved,
                "hpac_bytes": hpac_len,
                "semantic_bytes": semantic_len,
                "carrier_bytes": carrier_len,
            },
        },
    }

    nr1_measured_delta = sum(
        trials[name]["measured_substream_delta_bytes"] for name in ("nr1_qctx", "nr1_qpair", "nr1_qevent")
    )
    qpair_delta = trials["nr1_qpair"]["measured_substream_delta_bytes"]
    projected_ni1_archive = NI1_ARCHIVE.stat().st_size - qpair_delta
    fire_order = {
        "schema": "ddm.ad2.sealed_fire_order.v1",
        "basis": {
            "measured_qpair_substream_delta_bytes": qpair_delta,
            "current_ni1_archive": file_receipt(NI1_ARCHIVE),
            "projected_archive_bytes_if_isolated_delta_realizes": projected_ni1_archive,
            "projection_not_measurement": True,
        },
        "actions": [
            {
                "id": "ad2_qpair_tile_time_receiver_integration",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN assigns ddm_ni2_nr1_qpair_tile_time_receiver",
                "consumer_store": (
                    "/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/qpair_tile_time_receiver_r1/"
                ),
                "fire_trigger": (
                    "NI1 build_r4 scorer harvest is terminal or MAIN explicitly forks an isolated "
                    "successor; NI1 and AD2 pins revalidate; the shared staged index is empty"
                ),
                "action": (
                    "add a receiver-recognized tile-time QPAIR representation ID, rebuild the exact "
                    "NR1 packet and full archive, and require two byte-identical decodes matching "
                    "the current K32 token output before admitting any archive delta"
                ),
            },
            {
                "id": "ad2_qpair_tile_time_main_scorer",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN scorer-lane dispatcher",
                "consumer_store": (
                    "/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/"
                    "qpair_tile_time_receiver_r1/harvest/"
                ),
                "fire_trigger": (
                    "the integration result holds a byte-closed archive, repeat-identical full-RGB "
                    "decode, exact equality to NI1 K32 output, and MAIN holds the sole n600 slot"
                ),
                "action": "run the governed advisory/per-class repeats on the exact integrated archive",
            },
        ],
    }
    fire_order_receipt = retainer.json("SEALED_FIRE_ORDER.json", fire_order)
    result = {
        "schema": "ddm.ad2.addressing_cost_decomposition.v1",
        "axis": "[macOS-CPU scorer-free retained-receipt representation measurement n600]",
        "score_claim": False,
        "frontier_moved": False,
        "scope": (
            "byte-only decomposition, retained empirical entropy-model bounds, exact lossless "
            "substream transforms, "
            "and receiver ambiguity witnesses; no scorer and no live-candidate integration"
        ),
        "storage": storage,
        "pins": pins,
        "inherited_numbers": {
            "dc1_position_bytes": dc1_position_bits / 8,
            "dc1_length_bytes": dc1_length_bits / 8,
            "rc1_assignment_bytes": int(rc1_assignment_bytes),
            "rc1_assignment_fraction": int(rc1_assignment_bytes) / rc1_total,
            "nr1_qpair_physical_bytes": nr1.physical_attribution(nr1_packet)[nr1.Section.QPAIR][1]
            - nr1.physical_attribution(nr1_packet)[nr1.Section.QPAIR][0],
            "nr1_qpair_physical_fraction": (
                nr1.physical_attribution(nr1_packet)[nr1.Section.QPAIR][1]
                - nr1.physical_attribution(nr1_packet)[nr1.Section.QPAIR][0]
            )
            / nr1_total,
        },
        "decompositions": decompositions,
        "entropy": entropy,
        "representation_trials": trials,
        "ambiguity_witnesses": ambiguity,
        "sealed_fire_order": {"receipt": fire_order_receipt, **fire_order},
        "receiver_derivability": {
            "rc1_assignment": {
                "generic_free_rule": "raster coordinate order is implicit from header height and width",
                "nonderived_video_state": "the assignment ID at each coordinate",
                "rule118": "assignment IDs remain counted",
            },
            "nr1_qctx": {
                "generic_free_rule": "tile addresses are raster-derived from QCTX gh and gw",
                "nonderived_video_state": "the baseline dictionary ID at each tile",
                "rule118": "baseline IDs remain counted",
            },
            "nr1_qpair": {
                "generic_free_rule": (
                    "pair/tile addresses and the meanings previous/baseline/direct are fixed by shape"
                ),
                "nonderived_video_state": "the choice symbol for every pair and tile",
                "rule118": "choice symbols remain counted",
            },
            "nr1_qevent": {
                "generic_free_rule": "sorted delta-ULEB expansion is generic",
                "nonderived_video_state": "which coordinates are corrected and their class values",
                "rule118": "both coordinates and values remain counted",
            },
            "dx2_hpac": {
                "generic_free_rule": "the HPAC parser/decoder is generic",
                "nonderived_video_state": (
                    "the learned probability table is required before token decoding, so tokens cannot "
                    "derive it without circularity"
                ),
                "rule118": "HPAC table remains counted",
            },
        },
        "ranked_measured_levers": {
            "rc1": [
                {
                    "rank": 1,
                    "lever": "tested lossless assignment representations",
                    "measured_substream_delta_bytes": trials["rc1_assignment"]["measured_substream_delta_bytes"],
                    "disposition": "CLOSED at INSTANCE scope for tested lossless layouts",
                },
                {
                    "rank": 2,
                    "lever": "codebook-side work",
                    "measured_delta_bytes": None,
                    "disposition": "OWNED by CB2; AD2 does not touch it",
                },
            ],
            "nr1": [
                {
                    "rank": 1,
                    "lever": "lossless QPAIR reorder/bitpack",
                    "measured_substream_delta_bytes": trials["nr1_qpair"]["measured_substream_delta_bytes"],
                    "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                },
                {
                    "rank": 2,
                    "lever": "lossless QEVENT split layout",
                    "measured_substream_delta_bytes": trials["nr1_qevent"]["measured_substream_delta_bytes"],
                    "disposition": "FOLDED: measured 0 B",
                },
                {
                    "rank": 3,
                    "lever": "lossless QCTX reorder/bitpack",
                    "measured_substream_delta_bytes": trials["nr1_qctx"]["measured_substream_delta_bytes"],
                    "disposition": "FOLDED: measured 0 B",
                },
            ],
            "dx2": [
                {
                    "rank": 1,
                    "lever": "existing fixed-representation streams",
                    "measured_archive_delta_bytes": 0,
                    "disposition": "CLOSED by RB1; not reopened",
                }
            ],
        },
        "prior_law": {
            "prediction": (
                "RC1 is codebook-side, NR1 is addressing-side, and at least one NR1 addressing "
                "stream has at least a 20% gap to its declared context-model plug-in bound"
            ),
            "rc1_addressing_fraction": int(rc1_assignment_bytes) / rc1_total,
            "nr1_qpair_physical_fraction": 52_124 / nr1_total,
            "nr1_at_least_one_20pct_context_model_bound_gap": any(
                entropy[name].get("incumbent_gap_to_context_model_bound_fraction", 0) >= 0.20
                for name in ("nr1_qctx", "nr1_qpair")
            ),
            "verdict_scope": "INSTANCE: exact retained RC1-K2048 and NR1-K32 packets only",
            "measured_nr1_combined_substream_delta_upper_bound_bytes": nr1_measured_delta,
            "composition_warning": "sum is an upper bound; UNION is not SUM OF LEGS",
        },
        "rate_context": {
            "dx2_score": 0.14821987563243377,
            "dx2_bytes": 180_368,
            "strict_sub012_ceiling_bytes": 137_986,
            "cut_required_bytes": 42_382,
            "rate_score_per_byte": RATE_PER_BYTE,
            "rc1_under_ceiling_bytes": 24_980,
            "nr1_under_ceiling_bytes": 2_391,
            "ni1_landed_archive_bytes": NI1_ARCHIVE.stat().st_size,
            "ni1_projected_bytes_if_qpair_delta_realizes": projected_ni1_archive,
            "ni1_projection_headroom_below_ceiling_bytes": 137_986 - projected_ni1_archive,
            "projected_rate_score_reduction": qpair_delta * RATE_PER_BYTE,
            "projection_warning": "not an integrated archive and not a score",
        },
        "retention": {
            "artifacts": retainer.artifacts,
            "policy": "all materialized payloads and losing coder variants retained; no deletion",
        },
    }
    retainer.json("stage_02_measurement_complete.json", result)
    result["retention"]["artifacts"] = retainer.artifacts
    retainer.json("RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="completed AD2 output root; verifies and returns its RESULT.json",
    )
    parser.add_argument(
        "--cold-store-superseded-to",
        type=Path,
        help="certify and move completed measurement_v2..v5, keeping measurement_v6 local",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cold_store_superseded_to is not None:
        execution = cold_store_superseded_measurements(
            args.output_root.parent,
            args.output_root,
            args.cold_store_superseded_to,
        )
        print(json.dumps(execution, sort_keys=True))
        return
    if args.resume_from is not None:
        result_path = args.resume_from / "RESULT.json"
        if not result_path.is_file():
            raise SystemExit(f"resume root has no completed RESULT.json: {args.resume_from}")
        result = json.loads(result_path.read_text())
        if result.get("schema") != "ddm.ad2.addressing_cost_decomposition.v1":
            raise SystemExit("resume RESULT schema differs")
        for artifact in result["retention"]["artifacts"]:
            path = Path(artifact["path"])
            if file_receipt(path) != artifact:
                raise SystemExit(f"resume artifact drifted: {path}")
        print(
            json.dumps(
                {"resumed": str(result_path), "verified_artifacts": len(result["retention"]["artifacts"])},
                sort_keys=True,
            )
        )
        return
    started = time.time()
    result = run(args.output_root)
    print(
        json.dumps(
            {
                "result": str(args.output_root / "RESULT.json"),
                "artifacts": len(result["retention"]["artifacts"]),
                "elapsed_seconds": time.time() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
