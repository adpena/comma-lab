#!/usr/bin/env python3
"""Lossless model-section autopsy and retained real-coder race for DDM MZ1.

The input is the exact RX2 e480b winner model section.  Every materialized
coder payload is retained before its size is recorded.  Candidate containers
have a strict decoder in this file and are compared as complete framed model
sections, never as entropy ideals.  This runner is scorer-free.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

rx1 = importlib.import_module("experiments.ddm_rx1_rate_representation_attack")
rx2 = importlib.import_module("experiments.ddm_rx2_mc36_identity_race")
cp = importlib.import_module("experiments.ddm_cp135_rate_compose")

OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race")
RX2_ROOT = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac")
SOURCE_FINAL = RX2_ROOT / "FINAL_RESULT.json"
SOURCE_MODEL = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/models.rx1m"
SOURCE_MEMBER = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/p"
SOURCE_ARCHIVE = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/archive.zip"
SOURCE_REPEAT = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/archive.repeat.zip"
SOURCE_TOKEN = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/tokens.rc64"
SOURCE_RESIDUAL = RX2_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10/residual.compact.bin"
SOURCE_CHECKPOINT = (
    RX2_ROOT
    / "gpu_race/full_e480b/checkpoints/full_mps_e480.checkpoints/qat_stage_end_epoch_0480.pt"
)
SOURCE_RUN_LOG = RX2_ROOT / "gpu_race/full_e480b/launcher/run.log"
SOURCE_CPU_RECEIPT = (
    RX2_ROOT / "retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json"
)
EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")

EXPECTED = {
    "archive": (183_502, "e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3"),
    "member": (183_402, "30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58"),
    "model": (70_557, "7cf390160189e8708faf3a7b09a76fc18cee85e45fdc7f71d30f725014417411"),
    "token": (112_749, "b981b8399f184795da7cd99b8ee44416bd672c8c4ed1672f1252b32a64c10627"),
    "residual": (96, "64bbf9dfd88d6eb50d111f72d968ab7e8f8dc0ab00fb675d8ed2ee8a410b73ac"),
    "checkpoint": (1_099_767, "cd89907b5330bd78f9c1477107504231792c235fa7637b8981698a10948a5a61"),
}
EXPECTED_IHS1 = (17_996, "94526d667a9c8b98f1e3ef8d39fe8769d6cc6721cb9a102629ad47f26016460d")
EXPECTED_CPU_RAW = (3_662_409_600, "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9")
TRAINER_MODEL_ESTIMATE = 17_991
REQUIRED_SAVINGS = 15_153
RATE_DENOMINATOR = 37_545_489
AUTH_SCORE = 0.1600920261571558
AXIS = "[macOS-CPU advisory, scorer-free lossless model-section race]"
SCORE_CLAIM = False

CONTAINER_PREFIX = struct.Struct("<4sBBH")
CONTAINER_ENTRY = struct.Struct("<BBII")
CONTAINER_MAGIC = b"MZC1"
CONTAINER_VERSION = 1
BYTE_SEGMENT = 0
BIT_SEGMENT = 1

Codec = Literal["identity", "brotli_q11", "raw_lzma2", "rc64_adaptive", "smevr_r7", "byte_map_q11"]
CODEC_IDS: dict[Codec, int] = {
    "identity": 0,
    "brotli_q11": 1,
    "raw_lzma2": 2,
    "rc64_adaptive": 3,
    "smevr_r7": 4,
    "byte_map_q11": 5,
}
ID_CODECS = {value: key for key, value in CODEC_IDS.items()}


class MZ1Error(RuntimeError):
    """Fail-closed MZ1 input, framing, or identity error."""


@dataclass(frozen=True)
class LogicalSegment:
    name: str
    kind: int
    units: int
    data: bytes


@dataclass(frozen=True)
class EncodedSegment:
    source: LogicalSegment
    codec: Codec
    payload: bytes


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MZ1Error(f"retained payload absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def require_file(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    record = file_record(path)
    if expected is not None and (record["bytes"], record["sha256"]) != expected:
        raise MZ1Error(f"input pin changed: {path}")
    return record


def preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < 64 << 20:
        raise MZ1Error(f"MZ1 requires 64 MiB free at {output}; observed {free}")
    final = json.loads(SOURCE_FINAL.read_text(encoding="utf-8"))
    if final.get("schema") != "ddm_rx2_final.v1" or final.get("winner", {}).get("archive", {}).get("sha256") != EXPECTED["archive"][1]:
        raise MZ1Error("RX2 final receipt or winner pin changed")
    pins = {
        "rx2_final_result": require_file(SOURCE_FINAL),
        "archive": require_file(SOURCE_ARCHIVE, EXPECTED["archive"]),
        "repeat_archive": require_file(SOURCE_REPEAT, EXPECTED["archive"]),
        "member": require_file(SOURCE_MEMBER, EXPECTED["member"]),
        "model": require_file(SOURCE_MODEL, EXPECTED["model"]),
        "token": require_file(SOURCE_TOKEN, EXPECTED["token"]),
        "residual": require_file(SOURCE_RESIDUAL, EXPECTED["residual"]),
        "checkpoint": require_file(SOURCE_CHECKPOINT, EXPECTED["checkpoint"]),
        "launcher_run_log": require_file(SOURCE_RUN_LOG),
        "cpu_identity_receipt": require_file(SOURCE_CPU_RECEIPT),
    }
    result = {
        "schema": "ddm_mz1_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "storage": {"root": str(output.resolve()), "free_bytes": free, "required_bytes": 64 << 20},
        "inputs": pins,
    }
    atomic_json(output / "PREFLIGHT.json", result)
    return result


def _brotli(value: bytes, quality: int) -> bytes:
    completed = subprocess.run(["brotli", "-q", str(quality), "-c"], input=value, capture_output=True, check=False)
    if completed.returncode:
        raise MZ1Error(f"Brotli q{quality} failed: {completed.stderr.decode(errors='replace')}")
    restored = subprocess.run(["brotli", "-d", "-c"], input=completed.stdout, capture_output=True, check=False)
    if restored.returncode or restored.stdout != value:
        raise MZ1Error(f"Brotli q{quality} parse-back differs")
    return completed.stdout


def _raw_lzma2(value: bytes) -> bytes:
    payload = lzma.compress(value, format=lzma.FORMAT_RAW, filters=cp.LZMA_FILTERS)
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=cp.LZMA_FILTERS)
    restored = decoder.decompress(payload)
    if not decoder.eof or decoder.unused_data or restored != value:
        raise MZ1Error("raw LZMA2 parse-back differs")
    return payload


def _load_sections() -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    model = SOURCE_MODEL.read_bytes()
    if len(model) < rx1.RX1_HEADER.size:
        raise MZ1Error("RX1 model section is truncated")
    magic, version, codec_id, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = rx1.RX1_HEADER.unpack_from(model)
    if (magic, version, codec_id, table_mode, reserved) != (b"RX1M", 1, rx1.RX1_CODEC_BROTLI, rx1.RX1_TABLE_ON, 0):
        raise MZ1Error("RX1 model-section header changed")
    if rx1.RX1_HEADER.size + hpac_bytes + semantic_bytes + carrier_bytes != len(model):
        raise MZ1Error("RX1 model-section field accounting differs")
    offset = rx1.RX1_HEADER.size
    streams = []
    for size in (hpac_bytes, semantic_bytes, carrier_bytes):
        streams.append(model[offset : offset + size])
        offset += size
    raw = tuple(rx1._brotli_restore(stream, "brotli") for stream in streams)
    if (len(raw[0]), sha256_bytes(raw[0])) != EXPECTED_IHS1:
        raise MZ1Error("canonical e480b IHS1 bytes changed")
    return raw[0], raw[1], raw[2], {
        "header_bytes": rx1.RX1_HEADER.size,
        "codec": "split_brotli",
        "codec_qualities": {"hpac": 10, "semantic": 11, "carrier": 11},
        "compressed_bytes": dict(zip(("hpac", "semantic", "carrier"), (hpac_bytes, semantic_bytes, carrier_bytes), strict=True)),
        "compressed_sha256": dict(zip(("hpac", "semantic", "carrier"), map(sha256_bytes, streams), strict=True)),
    }


def _ihs1_layout(raw: bytes) -> tuple[dict[str, Any], list[LogicalSegment]]:
    if not raw.startswith(b"IHS1"):
        raise MZ1Error("HPAC raw section is not IHS1")
    packer = rx2._import_packer()
    model = packer.model_from_args(rx2._pack_args(), False).eval()
    modules = packer.compressible_modules(model)
    module_names = {id(module): name for name, module in model.named_modules()}
    channel_count = sum(int(module.weight.shape[0]) for module in modules)
    metadata_bytes = (channel_count + 1) // 2
    depths, _ = packer.unpack_nibbles(memoryview(raw)[4:], channel_count)
    depth_offset = 0
    module_rows = []
    total_weight_bits = 0
    for module in modules:
        module_depths = depths[depth_offset : depth_offset + module.weight.shape[0]]
        row_counts = [len(row) for row in packer.module_weight_rows(module, module.weight)]
        bits = sum(int(depth) * count for depth, count in zip(module_depths, row_counts, strict=True))
        module_rows.append({
            "name": module_names[id(module)],
            "tensor_shape": list(module.weight.shape),
            "output_channels": int(module.weight.shape[0]),
            "nonmasked_values": sum(row_counts),
            "packed_weight_bits": bits,
            "packed_weight_byte_equivalent": bits / 8.0,
            "depth_histogram": {str(value): int(np.count_nonzero(module_depths == value)) for value in np.unique(module_depths)},
        })
        total_weight_bits += bits
        depth_offset += module.weight.shape[0]
    weight_bytes = (total_weight_bits + 7) // 8
    metadata = raw[4 : 4 + metadata_bytes]
    weight_plane = raw[4 + metadata_bytes : 4 + metadata_bytes + weight_bytes]
    fixed = raw[4 + metadata_bytes + weight_bytes :]
    all_bits = np.unpackbits(np.frombuffer(weight_plane, dtype=np.uint8), bitorder="little")[:total_weight_bits]
    segments = [
        LogicalSegment("ihs1_magic", BYTE_SEGMENT, 4, b"IHS1"),
        LogicalSegment("deployed_depth_metadata", BYTE_SEGMENT, len(metadata), metadata),
    ]
    bit_offset = 0
    for row in module_rows:
        count = int(row["packed_weight_bits"])
        payload = np.packbits(all_bits[bit_offset : bit_offset + count], bitorder="little").tobytes()
        segments.append(LogicalSegment(f"weight::{row['name']}", BIT_SEGMENT, count, payload))
        bit_offset += count

    module_by_name = dict(model.named_modules())
    fixed_rows = []
    fixed_offset = 0
    for name, parameter in model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = module_by_name[module_name]
        if field == "weight" and isinstance(module, packer.COMPRESSIBLE_TYPES):
            continue
        dtype = np.dtype("<i2" if field == "bias" else "i1")
        size = parameter.numel() * dtype.itemsize
        value = fixed[fixed_offset : fixed_offset + size]
        if len(value) != size:
            raise MZ1Error(f"fixed IHS1 tensor is truncated: {name}")
        fixed_rows.append({"name": name, "shape": list(parameter.shape), "dtype": dtype.str, "bytes": size})
        segments.append(LogicalSegment(f"fixed::{name}", BYTE_SEGMENT, size, value))
        fixed_offset += size
    if bit_offset != total_weight_bits or fixed_offset != len(fixed):
        raise MZ1Error("IHS1 logical autopsy did not consume every bit and byte")
    rebuilt = _rebuild_ihs1(segments)
    if rebuilt != raw:
        raise MZ1Error("IHS1 logical segmentation is not byte-reversible")
    layout = {
        "raw_bytes": len(raw),
        "magic_bytes": 4,
        "deployed_depth_metadata_bytes": metadata_bytes,
        "deployed_depth_count": channel_count,
        "weight_plane_bits": total_weight_bits,
        "weight_plane_bytes": weight_bytes,
        "fixed_parameter_bytes": len(fixed),
        "module_tensor_count": len(module_rows),
        "fixed_tensor_count": len(fixed_rows),
        "module_tensors": module_rows,
        "fixed_tensors": fixed_rows,
        "shape_and_index_metadata_shipped_bytes": 0,
        "training_only_bit_depth_buffers_shipped": False,
        "training_only_bit_depth_buffer_count": 9,
        "training_only_note": "The nine fp32 training buffers are absent. The 259-byte learned deployed-depth vector is shipped and is not derivable from config or seed.",
    }
    return layout, segments


def _rebuild_ihs1(segments: list[LogicalSegment]) -> bytes:
    if len(segments) < 3 or segments[0].data != b"IHS1" or segments[1].kind != BYTE_SEGMENT:
        raise MZ1Error("IHS1 logical segment order differs")
    output = bytearray(segments[0].data + segments[1].data)
    weight_bits = []
    index = 2
    while index < len(segments) and segments[index].kind == BIT_SEGMENT:
        segment = segments[index]
        bits = np.unpackbits(np.frombuffer(segment.data, dtype=np.uint8), bitorder="little")[: segment.units]
        weight_bits.append(bits)
        index += 1
    if not weight_bits:
        raise MZ1Error("IHS1 logical layout contains no weight tensors")
    output.extend(np.packbits(np.concatenate(weight_bits), bitorder="little").tobytes())
    for segment in segments[index:]:
        if segment.kind != BYTE_SEGMENT or len(segment.data) != segment.units:
            raise MZ1Error("IHS1 fixed tensor framing differs")
        output.extend(segment.data)
    return bytes(output)


def autopsy(output: Path) -> dict[str, Any]:
    admission = preflight(output)
    hpac, semantic, carrier, outer = _load_sections()
    layout, _ = _ihs1_layout(hpac)
    current = EXPECTED["model"][0]
    frozen_renderer_and_header = outer["header_bytes"] + outer["compressed_bytes"]["semantic"] + outer["compressed_bytes"]["carrier"]
    result = {
        "schema": "ddm_mz1_section_autopsy.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "preflight": admission,
        "current_model_section": require_file(SOURCE_MODEL, EXPECTED["model"]),
        "outer": outer,
        "raw_sections": {
            "hpac_ihs1": {"bytes": len(hpac), "sha256": sha256_bytes(hpac)},
            "semantic_physical": {"bytes": len(semantic), "sha256": sha256_bytes(semantic)},
            "carrier_physical": {"bytes": len(carrier), "sha256": sha256_bytes(carrier)},
        },
        "ihs1_layout": layout,
        "trainer_comparison": {
            "trainer_estimated_model_bytes": TRAINER_MODEL_ESTIMATE,
            "actual_ihs1_raw_bytes": len(hpac),
            "actual_ihs1_raw_minus_estimate": len(hpac) - TRAINER_MODEL_ESTIMATE,
            "actual_shipped_ihs1_compressed_bytes": outer["compressed_bytes"]["hpac"],
            "shipped_ihs1_minus_estimate": outer["compressed_bytes"]["hpac"] - TRAINER_MODEL_ESTIMATE,
            "frozen_semantic_carrier_plus_wrapper_bytes": frozen_renderer_and_header,
            "apparent_gap_model_section_minus_trainer_estimate": current - TRAINER_MODEL_ESTIMATE,
            "attribution_identity": f"{current} = {outer['compressed_bytes']['hpac']} HPAC + {outer['compressed_bytes']['semantic']} semantic + {outer['compressed_bytes']['carrier']} carrier + {outer['header_bytes']} wrapper",
            "verdict": "NO_SERIALIZATION_GAP: the 52,566-byte difference compares one HPAC estimate with a three-object model section; HPAC itself ships 4,372 bytes below the estimate.",
        },
        "container_fact_correction": "The winner uses split Brotli q10/q11/q11, not an XZ model container.",
        "all_payloads_retained": True,
    }
    raw_root = output / "retained/autopsy"
    atomic_bytes(raw_root / "hpac.ihs1.raw", hpac)
    atomic_bytes(raw_root / "semantic.physical.raw", semantic)
    atomic_bytes(raw_root / "carrier.physical.raw", carrier)
    atomic_json(output / "AUTOPSY_RESULT.json", result)
    return result


def _rc64_classes(output: Path) -> tuple[Any, Any, Path]:
    library = cp._compile_checkpointable_rc64(SimpleNamespace(experiment_book=EXPERIMENT_BOOK, output=output))
    sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
    try:
        module = importlib.import_module("cpr1_sub4.entropy.rc64")
    finally:
        sys.path.pop(0)
    return module.NativeEncoder, module.NativeDecoder, library


def _base5_digits(value: bytes) -> np.ndarray:
    values = np.frombuffer(value, dtype=np.uint8).astype(np.uint16)
    digits = np.empty((len(values), 4), dtype=np.uint8)
    work = values.copy()
    for index in range(4):
        digits[:, index] = work % 5
        work //= 5
    return digits.reshape(-1)


def _adaptive_probabilities(symbols: np.ndarray) -> np.ndarray:
    probabilities = np.empty((len(symbols), 5), dtype=np.float32)
    counts = np.ones(5, dtype=np.float64)
    total = 5.0
    for index, symbol in enumerate(symbols):
        probabilities[index] = counts / total
        counts[int(symbol)] += 1.0
        total += 1.0
    return probabilities


def _rc64_encode(value: bytes, native_encoder: Any, library: Path) -> bytes:
    encoder = native_encoder(library)
    symbols = _base5_digits(value)
    encoder.encode(symbols.astype(np.int32), _adaptive_probabilities(symbols))
    return encoder.finish()


def _rc64_decode(payload: bytes, count: int, native_decoder: Any, library: Path) -> bytes:
    decoder = native_decoder(library, payload)
    counts = np.ones(5, dtype=np.float64)
    total = 5.0
    digits = np.empty(count * 4, dtype=np.uint8)
    for index in range(len(digits)):
        probability = (counts / total).astype(np.float32)[None, :]
        symbol = int(decoder.decode(probability)[0])
        digits[index] = symbol
        counts[symbol] += 1.0
        total += 1.0
    rows = digits.reshape(count, 4).astype(np.uint16)
    values = rows[:, 0] + 5 * rows[:, 1] + 25 * rows[:, 2] + 125 * rows[:, 3]
    if np.any(values > 255):
        raise MZ1Error("RC64 base-5 stream decoded a non-byte code")
    return values.astype(np.uint8).tobytes()


def _byte_map_encode(value: bytes) -> bytes:
    array = np.frombuffer(value, dtype=np.uint8)
    counts = np.bincount(array, minlength=256)
    used = np.flatnonzero(counts)
    order = np.asarray(sorted(used.tolist(), key=lambda item: (-int(counts[item]), int(item))), dtype=np.uint8)
    inverse = np.zeros(256, dtype=np.uint8)
    inverse[order] = np.arange(len(order), dtype=np.uint8)
    mapped = inverse[array].tobytes()
    return struct.pack("<H", len(order)) + order.tobytes() + _brotli(mapped, 11)


def _byte_map_decode(payload: bytes, count: int) -> bytes:
    if len(payload) < 2:
        raise MZ1Error("byte-map payload is truncated")
    width = struct.unpack_from("<H", payload)[0]
    if not 1 <= width <= 256 or len(payload) < 2 + width:
        raise MZ1Error("byte-map dictionary differs")
    order = np.frombuffer(payload[2 : 2 + width], dtype=np.uint8)
    mapped = rx1._brotli_restore(payload[2 + width :], "brotli")
    indices = np.frombuffer(mapped, dtype=np.uint8)
    if len(indices) != count or np.any(indices >= width):
        raise MZ1Error("byte-map index stream differs")
    return order[indices].tobytes()


def _encode(codec: Codec, value: bytes, rc64: tuple[Any, Any, Path]) -> bytes:
    native_encoder, _, library = rc64
    if codec == "identity":
        return value
    if codec == "brotli_q11":
        return _brotli(value, 11)
    if codec == "raw_lzma2":
        return _raw_lzma2(value)
    if codec == "rc64_adaptive":
        return _rc64_encode(value, native_encoder, library)
    if codec == "smevr_r7":
        return cp._smevr_roundtrip(value)
    if codec == "byte_map_q11":
        return _byte_map_encode(value)
    raise MZ1Error(f"unsupported codec: {codec}")


def _decode(codec: Codec, payload: bytes, count: int, rc64: tuple[Any, Any, Path]) -> bytes:
    _, native_decoder, library = rc64
    if codec == "identity":
        return payload
    if codec == "brotli_q11":
        return rx1._brotli_restore(payload, "brotli")
    if codec == "raw_lzma2":
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=cp.LZMA_FILTERS)
        value = decoder.decompress(payload)
        if not decoder.eof or decoder.unused_data:
            raise MZ1Error("raw LZMA2 candidate is truncated or has a trailer")
        return value
    if codec == "rc64_adaptive":
        return _rc64_decode(payload, count, native_decoder, library)
    if codec == "smevr_r7":
        import ddm_bd1_class_field_receiver as bd1
        return b"".join(bd1.unsmevr_records(payload))
    if codec == "byte_map_q11":
        return _byte_map_decode(payload, count)
    raise MZ1Error(f"unsupported codec: {codec}")


def _pack_container(segments: list[EncodedSegment]) -> bytes:
    header = [CONTAINER_PREFIX.pack(CONTAINER_MAGIC, CONTAINER_VERSION, 0, len(segments))]
    for segment in segments:
        header.append(CONTAINER_ENTRY.pack(CODEC_IDS[segment.codec], segment.source.kind, segment.source.units, len(segment.payload)))
    return b"".join(header) + b"".join(segment.payload for segment in segments)


def _unpack_container(value: bytes, rc64: tuple[Any, Any, Path]) -> list[LogicalSegment]:
    if len(value) < CONTAINER_PREFIX.size:
        raise MZ1Error("candidate container is truncated")
    magic, version, reserved, count = CONTAINER_PREFIX.unpack_from(value)
    if (magic, version, reserved) != (CONTAINER_MAGIC, CONTAINER_VERSION, 0) or count == 0:
        raise MZ1Error("candidate container header differs")
    directory_end = CONTAINER_PREFIX.size + count * CONTAINER_ENTRY.size
    if directory_end > len(value):
        raise MZ1Error("candidate directory is truncated")
    cursor = directory_end
    restored = []
    for index in range(count):
        codec_id, kind, units, size = CONTAINER_ENTRY.unpack_from(value, CONTAINER_PREFIX.size + index * CONTAINER_ENTRY.size)
        if codec_id not in ID_CODECS or kind not in (BYTE_SEGMENT, BIT_SEGMENT) or cursor + size > len(value):
            raise MZ1Error("candidate directory entry differs")
        codec = ID_CODECS[codec_id]
        payload = value[cursor : cursor + size]
        cursor += size
        byte_count = units if kind == BYTE_SEGMENT else (units + 7) // 8
        decoded = _decode(codec, payload, byte_count, rc64)
        if len(decoded) != byte_count:
            raise MZ1Error("candidate segment decoded length differs")
        restored.append(LogicalSegment(f"segment_{index}", kind, units, decoded))
    if cursor != len(value):
        raise MZ1Error("candidate container has trailing bytes")
    return restored


def _retained_candidate(
    output: Path,
    name: str,
    sources: list[LogicalSegment],
    codecs: list[Codec],
    rc64: tuple[Any, Any, Path],
    *,
    ihs1_logical_count: int | None = None,
    split_whole: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if len(sources) != len(codecs):
        raise MZ1Error("candidate source/codec counts differ")
    root = output / "retained/candidates" / name
    encoded = []
    rows = []
    for index, (source, codec) in enumerate(zip(sources, codecs, strict=True)):
        payload = _encode(codec, source.data, rc64)
        path = root / "segments" / f"{index:03d}_{source.name}.{codec}.bin"
        atomic_bytes(path, payload)
        restored = _decode(codec, payload, len(source.data), rc64)
        if restored != source.data:
            raise MZ1Error(f"{name}/{source.name}/{codec} parse-back differs")
        encoded.append(EncodedSegment(source, codec, payload))
        rows.append({"name": source.name, "kind": source.kind, "units": source.units, "codec": codec, "source_bytes": len(source.data), "payload": file_record(path)})
    container = _pack_container(encoded)
    container_path = root / "model_section.mzc1"
    atomic_bytes(container_path, container)
    decoded = _unpack_container(container, rc64)
    if split_whole is not None:
        if len(decoded) != 1:
            raise MZ1Error("whole-section candidate decoded segment count differs")
        joined = decoded[0].data
        a, b, c = split_whole
        sections = (joined[:a], joined[a : a + b], joined[a + b : a + b + c])
    elif ihs1_logical_count is not None:
        hpac = _rebuild_ihs1(decoded[:ihs1_logical_count])
        if len(decoded) != ihs1_logical_count + 2:
            raise MZ1Error("per-tensor candidate section count differs")
        sections = (hpac, decoded[-2].data, decoded[-1].data)
    else:
        if len(decoded) != 3:
            raise MZ1Error("per-section candidate decoded segment count differs")
        sections = tuple(segment.data for segment in decoded)
    return {
        "name": name,
        "model_section": file_record(container_path),
        "segments": rows,
        "decoded_section_sha256": list(map(sha256_bytes, sections)),
        "decoded_section_bytes": list(map(len, sections)),
        "parseback_exact": True,
        "receiver_status": "measurement_decoder_closed; shipping RX1 receiver adapter required if selected",
    }


def race(output: Path) -> dict[str, Any]:
    if not (output / "AUTOPSY_RESULT.json").is_file():
        autopsy(output)
    hpac, semantic, carrier, _ = _load_sections()
    _, logical = _ihs1_layout(hpac)
    native_encoder, native_decoder, library = _rc64_classes(output)
    rc64 = (native_encoder, native_decoder, library)
    section_sources = [
        LogicalSegment("hpac_ihs1", BYTE_SEGMENT, len(hpac), hpac),
        LogicalSegment("semantic_physical", BYTE_SEGMENT, len(semantic), semantic),
        LogicalSegment("carrier_physical", BYTE_SEGMENT, len(carrier), carrier),
    ]
    expected_hashes = list(map(sha256_bytes, (hpac, semantic, carrier)))
    candidates = []
    current_root = output / "retained/candidates/current_rx1m"
    atomic_bytes(current_root / "model_section.rx1m", SOURCE_MODEL.read_bytes())
    candidates.append({
        "name": "current_rx1m",
        "model_section": file_record(current_root / "model_section.rx1m"),
        "segments": [],
        "decoded_section_sha256": expected_hashes,
        "decoded_section_bytes": list(map(len, (hpac, semantic, carrier))),
        "parseback_exact": True,
        "receiver_status": "shipping_receiver_closed",
    })
    for name, codec in (
        ("per_section_brotli_q11", "brotli_q11"),
        ("per_section_raw_lzma2", "raw_lzma2"),
        ("per_section_rc64_adaptive", "rc64_adaptive"),
        ("per_section_smevr_r7", "smevr_r7"),
    ):
        candidates.append(_retained_candidate(output, name, section_sources, [codec] * 3, rc64))
    joined = hpac + semantic + carrier
    joined_source = [LogicalSegment("all_raw_sections", BYTE_SEGMENT, len(joined), joined)]
    for name, codec in (("whole_brotli_q11", "brotli_q11"), ("whole_raw_lzma2", "raw_lzma2")):
        candidates.append(
            _retained_candidate(output, name, joined_source, [codec], rc64, split_whole=(len(hpac), len(semantic), len(carrier)))
        )

    tensor_sources = logical + section_sources[1:]
    mixed_codecs: list[Codec] = []
    mixed_rows = []
    trial_root = output / "retained/candidates/per_tensor_mixed/trials"
    for index, source in enumerate(tensor_sources):
        trials = []
        for codec in ("identity", "brotli_q11", "raw_lzma2", "rc64_adaptive", "byte_map_q11"):
            payload = _encode(codec, source.data, rc64)
            path = trial_root / f"{index:03d}_{source.name}" / f"{codec}.bin"
            atomic_bytes(path, payload)
            if _decode(codec, payload, len(source.data), rc64) != source.data:
                raise MZ1Error(f"per-tensor trial parse-back differs: {source.name}/{codec}")
            trials.append({"codec": codec, "payload": file_record(path)})
        winner = min(trials, key=lambda row: (row["payload"]["bytes"], row["codec"]))
        mixed_codecs.append(winner["codec"])
        mixed_rows.append({"segment": source.name, "denominator": len(trials), "trials": trials, "winner": winner})
    mixed = _retained_candidate(
        output,
        "per_tensor_mixed",
        tensor_sources,
        mixed_codecs,
        rc64,
        ihs1_logical_count=len(logical),
    )
    mixed["per_tensor_trials"] = mixed_rows
    mixed["selection"] = "minimum retained framed segment bytes; lexical codec breaks ties"
    candidates.append(mixed)

    for candidate in candidates:
        if candidate["decoded_section_sha256"] != expected_hashes:
            raise MZ1Error(f"candidate section parse-back differs: {candidate['name']}")
        candidate["delta_vs_current_model_section"] = candidate["model_section"]["bytes"] - EXPECTED["model"][0]
        candidate["exact_savings_vs_current"] = -candidate["delta_vs_current_model_section"]
        candidate["meets_15153_byte_bar"] = candidate["exact_savings_vs_current"] >= REQUIRED_SAVINGS
    winner = min(candidates, key=lambda row: (row["model_section"]["bytes"], row["name"]))
    result = {
        "schema": "ddm_mz1_model_section_race.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "source_model_section": require_file(SOURCE_MODEL, EXPECTED["model"]),
        "candidates": candidates,
        "candidate_denominator": len(candidates),
        "winner": winner,
        "required_savings_for_sub015": REQUIRED_SAVINGS,
        "winner_meets_required_savings": winner["meets_15153_byte_bar"],
        "all_payloads_retained": True,
        "rc64_source": file_record(output / "work/rc64_checkpoint_backend.c"),
        "rc64_library": file_record(library),
        "selection": "minimum complete retained model-section bytes; lexical name breaks ties",
        "verdict_scope": "INSTANCE: exact e480b RX2 model section; lossless same-decoded-section forms",
    }
    atomic_json(output / "RACE_RESULT.json", result)
    return result


def build(output: Path) -> dict[str, Any]:
    race_path = output / "RACE_RESULT.json"
    if not race_path.is_file():
        race(output)
    measured = json.loads(race_path.read_text(encoding="utf-8"))
    winner = measured["winner"]
    if winner["name"] != "current_rx1m":
        result = {
            "schema": "ddm_mz1_build.v1",
            "complete": False,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
            "blocker": "Measured winner uses MZC1 and requires a shipping receiver adapter before an archive can be honestly built.",
            "winner": winner,
        }
        atomic_json(output / "BUILD_RESULT.json", result)
        return result
    root = output / "retained/winner"
    model = SOURCE_MODEL.read_bytes()
    residual = SOURCE_RESIDUAL.read_bytes()
    token = SOURCE_TOKEN.read_bytes()
    member = model + residual + token
    if member != SOURCE_MEMBER.read_bytes():
        raise MZ1Error("winner member rebuild differs from RX2 custody")
    archive = rx1.deterministic_zip(member)
    if archive != SOURCE_ARCHIVE.read_bytes() or archive != SOURCE_REPEAT.read_bytes():
        raise MZ1Error("winner deterministic archive rebuild differs from RX2 custody")
    records = {
        "model": atomic_bytes(root / "model_section.rx1m", model),
        "residual": atomic_bytes(root / "residual.compact.bin", residual),
        "token": atomic_bytes(root / "tokens.rc64", token),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "repeat_archive": atomic_bytes(root / "archive.repeat.zip", rx1.deterministic_zip(member)),
    }
    cpu = json.loads(SOURCE_CPU_RECEIPT.read_text(encoding="utf-8"))
    if (
        cpu.get("complete") is not True
        or cpu.get("candidate", {}).get("sha256") != records["archive"]["sha256"]
        or cpu.get("decoded_token_identity") is not True
        or cpu.get("raw_identity_vs_mc36_cpu") is not True
        or (cpu.get("raw_output", {}).get("bytes"), cpu.get("raw_output", {}).get("sha256")) != EXPECTED_CPU_RAW
    ):
        raise MZ1Error("existing CPU identity receipt does not bind the rebuilt exact archive")
    result = {
        "schema": "ddm_mz1_build.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "winner": winner,
        "payloads": records,
        "repeat_byte_identical": records["archive"]["sha256"] == records["repeat_archive"]["sha256"],
        "archive_byte_identical_to_rx2": records["archive"]["sha256"] == EXPECTED["archive"][1],
        "cpu_identity_transfer": {
            "basis": "The rebuilt archive is byte-identical to the already decoded CPU byte object.",
            "receipt": require_file(SOURCE_CPU_RECEIPT),
            "decoded_token_identity": True,
            "raw_identity_vs_mc36_cpu": True,
            "raw_output": cpu["raw_output"],
            "rerun_required": False,
        },
        "all_payloads_retained": True,
    }
    atomic_json(output / "BUILD_RESULT.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    if not (output / "BUILD_RESULT.json").is_file():
        build(output)
    autopsy_result = json.loads((output / "AUTOPSY_RESULT.json").read_text(encoding="utf-8"))
    race_result = json.loads((output / "RACE_RESULT.json").read_text(encoding="utf-8"))
    build_result = json.loads((output / "BUILD_RESULT.json").read_text(encoding="utf-8"))
    winner = race_result["winner"]
    savings = int(winner["exact_savings_vs_current"])
    projected = AUTH_SCORE - savings * 25 / RATE_DENOMINATOR
    t4_fire_order = {
        "schema": "ddm_mz1_t4_fire_order.v1",
        "disposition": (
            "QUEUED-WITH-A-FIRE-ORDER"
            if savings > 0 and build_result.get("complete") is True
            else "FOLDED"
        ),
        "owner": "MAIN",
        "consumer_store": str((output / "FINAL_RESULT.json").resolve()),
        "current_fire": False,
        "reason": (
            "A smaller byte-distinct archive is receiver-closed and ready for paired T4."
            if savings > 0 and build_result.get("complete") is True
            else "The winning archive is byte-identical to the existing authority row; duplicate paid evaluation is forbidden."
        ),
        "fire_trigger": (
            "A strictly smaller archive than 183502 bytes exists under custody, passes the shipping receiver, "
            "and proves exact token-stream plus decoded-raw identity before dispatch."
        ),
        "source_authority_call": "fc-01M02QMN3SQ9SNHXZMRWXYEJEW",
        "source_archive": require_file(SOURCE_ARCHIVE, EXPECTED["archive"]),
    }
    atomic_json(output / "T4_FIRE_ORDER.json", t4_fire_order)
    inventory_files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.name in {"FINAL_RESULT.json", "RETENTION_INVENTORY.json"}:
            continue
        inventory_files.append(file_record(path))
    inventory = {
        "schema": "ddm_mz1_retention_inventory.v1",
        "root": str(output.resolve()),
        "file_denominator": len(inventory_files),
        "retained_bytes": sum(int(item["bytes"]) for item in inventory_files),
        "files": inventory_files,
    }
    atomic_json(output / "RETENTION_INVENTORY.json", inventory)
    result = {
        "schema": "ddm_mz1_final.v1",
        "complete": build_result.get("complete") is True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "source_rx2_schema": "ddm_rx2_final.v1",
        "source_archive": require_file(SOURCE_ARCHIVE, EXPECTED["archive"]),
        "receipts": {
            "preflight": file_record(output / "PREFLIGHT.json"),
            "autopsy": file_record(output / "AUTOPSY_RESULT.json"),
            "race": file_record(output / "RACE_RESULT.json"),
            "build": file_record(output / "BUILD_RESULT.json"),
            "t4_fire_order": file_record(output / "T4_FIRE_ORDER.json"),
            "retention_inventory": file_record(output / "RETENTION_INVENTORY.json"),
            "source_cpu_identity": require_file(SOURCE_CPU_RECEIPT),
        },
        "winner": winner,
        "exact_model_section_savings": savings,
        "required_savings_for_sub015": REQUIRED_SAVINGS,
        "sub015_rate_rung_crossed": savings >= REQUIRED_SAVINGS,
        "projected_score_if_exact_rx2_distortion_held": projected,
        "projection_label": "PROJECTED_NOT_AUTHORITY; no scorer or evaluator ran in MZ1",
        "premise_verdict": autopsy_result["trainer_comparison"]["verdict"],
        "build": build_result,
        "t4_disposition": t4_fire_order,
        "all_payloads_retained": True,
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("autopsy", "race", "build", "finalize", "all"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("autopsy", "all"):
        result = autopsy(args.output)
    if args.stage in ("race", "all"):
        result = race(args.output)
    if args.stage in ("build", "all"):
        result = build(args.output)
    if args.stage in ("finalize", "all"):
        result = finalize(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
