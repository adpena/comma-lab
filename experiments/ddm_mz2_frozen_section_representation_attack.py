#!/usr/bin/env python3
"""Current-e480b frozen semantic-section representation attack for DDM MZ2.

This runner is scorer-free.  It decodes the exact F12 semantic body, proves
the fixed receiver consumes all 38 schema records, retains every materialized
tensor/payload, and compares real complete archives for exact-state fixed-
schema representations.  A current-state mixed-bit payload is retained only
as a byte-gated scorer-queue input; it is never promoted without an n600
receiver-closed score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import struct
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

mz1 = importlib.import_module("experiments.ddm_mz1_model_section_rate_race")
rx1 = importlib.import_module("experiments.ddm_rx1_rate_representation_attack")
sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")
sm3 = importlib.import_module("experiments.ddm_sm3_semantic_representation")

OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation")
BOOK_SRC = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src")
RUNTIME_DIR = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/"
    "micro35_candidate/adapted_runtime/cpr1"
)
CANONICAL_WANS = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/"
    "pr135/canonical/semantic.wans1"
)
F12_BODY = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/"
    "pr135/l2/semantic_wans_f12_body.bin"
)
LEGACY_RAW = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/"
    "pr135/decoded/semantic_pr130_layout.raw"
)
MZ1_FINAL = Path("/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/FINAL_RESULT.json")
SM3_RESULT = Path("/Volumes/VertigoDataTier/pact/ddm_sm3_20260810/final_v3/SM3_RESULT.json")
SM4_RESULT = Path("/Volumes/VertigoDataTier/pact/ddm_sm4_20260810/GRID_RESULT.json")

EXPECTED_F12 = (36_040, "b0d41ec904aca82f93f3c8bc68d0e48896ba08efdaa7a4a2ee204f002fc28ec8")
EXPECTED_CANONICAL = (36_051, "b489c73567046e64a1644eb1bca5cb5ed86d690f2f98f703e22424ab97505521")
EXPECTED_LEGACY = 40_252
EXPECTED_SEMANTIC_STREAM_BYTES = 34_763
EXPECTED_CARRIER_STREAM_BYTES = 22_161
CURRENT_ARCHIVE_BYTES = 183_502
CURRENT_SCORE = 0.1600920261571558
CURRENT_ARCHIVE_SHA256 = "e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3"
RATE_DENOMINATOR = 37_545_489
REQUIRED_SAVINGS = 15_153
SEED = 20260815
AXIS = "[macOS-CPU advisory; scorer-free current-e480b section representation]"

MZ2E_MAGIC = b"MZ2E"
MZ2E_VERSION = 1
MODE_DENSE = 0
MODE_SPARSE = 1
MODE_ROWDICT = 2
MODE_NAMES = {MODE_DENSE: "dense", MODE_SPARSE: "zero_sparse", MODE_ROWDICT: "row_dictionary"}


class MZ2Error(RuntimeError):
    """Fail-closed MZ2 input, identity, or custody error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MZ2Error(f"required retained file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_file(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    record = file_record(path)
    if expected is not None and (record["bytes"], record["sha256"]) != expected:
        raise MZ2Error(f"input pin changed: {path}")
    return record


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


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def h0_bytes(value: bytes) -> float:
    if not value:
        return 0.0
    counts = np.bincount(np.frombuffer(value, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(value)
    return float(-(probabilities * np.log2(probabilities)).sum() * len(value) / 8)


def _book_modules():
    if str(BOOK_SRC) not in sys.path:
        sys.path.insert(0, str(BOOK_SRC))
    baseline = importlib.import_module("cpr1_sub4.baseline")
    codec = importlib.import_module("cpr1_sub4.entropy.renderer_weight_codec")
    archive = importlib.import_module("cpr1_sub4.residual_archive")
    bits = importlib.import_module("cpr1_sub4.bits")
    return baseline, codec, archive, bits


def _runtime_module():
    if str(RUNTIME_DIR) not in sys.path:
        sys.path.insert(0, str(RUNTIME_DIR))
    name = "ddm_mz2_frozen_f26_inflate"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, RUNTIME_DIR / "inflate.py")
    if spec is None or spec.loader is None:
        raise MZ2Error("could not load the frozen F26 runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_records():
    baseline, codec, archive, _ = _book_modules()
    body = F12_BODY.read_bytes()
    canonical = codec.decode_f12_wans_body(body, archive.F13_WANS_STREAM_ORDER)
    if canonical != CANONICAL_WANS.read_bytes():
        raise MZ2Error("F12 body does not restore the pinned canonical WANS bytes")
    records = codec.decode_wans1(canonical)
    legacy = baseline.encode_legacy_w4(records)
    if legacy != LEGACY_RAW.read_bytes():
        raise MZ2Error("decoded current WANS state differs from the retained legacy-layout state")
    return records, canonical, legacy


def _source_streams() -> tuple[bytes, bytes, bytes]:
    model = mz1.SOURCE_MODEL.read_bytes()
    fields = rx1.RX1_HEADER.unpack_from(model)
    magic, version, codec_id, table_mode, reserved, hpac_size, semantic_size, carrier_size = fields
    if (magic, version, codec_id, table_mode, reserved) != (
        rx1.RX1_MAGIC,
        rx1.RX1_VERSION,
        rx1.RX1_CODEC_BROTLI,
        rx1.RX1_TABLE_ON,
        0,
    ):
        raise MZ2Error("current RX1 header changed")
    if (semantic_size, carrier_size) != (EXPECTED_SEMANTIC_STREAM_BYTES, EXPECTED_CARRIER_STREAM_BYTES):
        raise MZ2Error("current RX1 semantic/carrier stream lengths changed")
    offset = rx1.RX1_HEADER.size
    hpac = model[offset : offset + hpac_size]
    semantic = model[offset + hpac_size : offset + hpac_size + semantic_size]
    carrier = model[offset + hpac_size + semantic_size :]
    if len(carrier) != carrier_size:
        raise MZ2Error("current RX1 field accounting changed")
    return hpac, semantic, carrier


def preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    required = 64 << 20
    if free < required:
        raise MZ2Error(f"MZ2 requires {required} free bytes at {output}; observed {free}")
    mz1_final = json.loads(MZ1_FINAL.read_text(encoding="utf-8"))
    if mz1_final.get("complete") is not True:
        raise MZ2Error("MZ1 closure receipt is incomplete")
    sm3_result = json.loads(SM3_RESULT.read_text(encoding="utf-8"))
    sm4_result = json.loads(SM4_RESULT.read_text(encoding="utf-8"))
    if (
        sm3_result.get("schema") != "ddm_sm3_semantic_representation_result.v1"
        or len(sm3_result.get("candidates", [])) != 8
        or sm4_result.get("complete") is not True
    ):
        raise MZ2Error("SM3/SM4 recall receipt is incomplete")
    if len(sm4_result.get("cells", [])) != 410:
        raise MZ2Error("SM4 recall grid denominator changed")
    inputs = {
        "source_archive": require_file(mz1.SOURCE_ARCHIVE, mz1.EXPECTED["archive"]),
        "source_member": require_file(mz1.SOURCE_MEMBER, mz1.EXPECTED["member"]),
        "source_model": require_file(mz1.SOURCE_MODEL, mz1.EXPECTED["model"]),
        "source_token": require_file(mz1.SOURCE_TOKEN, mz1.EXPECTED["token"]),
        "source_residual": require_file(mz1.SOURCE_RESIDUAL, mz1.EXPECTED["residual"]),
        "f12_body": require_file(F12_BODY, EXPECTED_F12),
        "canonical_wans": require_file(CANONICAL_WANS, EXPECTED_CANONICAL),
        "legacy_layout": require_file(LEGACY_RAW),
        "mz1_closure": require_file(MZ1_FINAL),
        "sm3_result": require_file(SM3_RESULT),
        "sm4_grid": require_file(SM4_RESULT),
        "runtime_inflater": require_file(RUNTIME_DIR / "inflate.py"),
    }
    if inputs["legacy_layout"]["bytes"] != EXPECTED_LEGACY:
        raise MZ2Error("legacy-layout semantic byte count changed")
    result = {
        "schema": "ddm_mz2_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "seed": SEED,
        "storage": {"root": str(output.resolve()), "free_bytes": free, "required_bytes": required},
        "inputs": inputs,
        "scorer_slot_claimed": False,
        "scorer_policy": "charter does not claim the one-full-n600 slot; retain candidates and queue scoring",
        "live_e960_touched": False,
        "resumable_from_disk": True,
        "stage_checkpoints": ["PREFLIGHT.json", "AUTOPSY_RESULT.json", "EXACT_RESULT.json", "SCORE_GATE_RESULT.json", "FINAL_RESULT.json"],
    }
    atomic_json(output / "PREFLIGHT.json", result)
    return result


def _raw_record_payload(record, bits_module) -> bytes:
    if record.schema.is_fp16:
        return record.raw_fp16 or np.asarray(record.values, dtype="<f2").tobytes()
    return (record.raw_scales or np.asarray(record.scales, dtype="<f2").tobytes()) + bits_module.pack_signed(
        np.asarray(record.codes, dtype=np.int8).reshape(-1), 4
    )


def _matrix_stats(record) -> dict[str, Any] | None:
    if record.schema.is_fp16:
        return None
    codes = np.asarray(record.codes, dtype=np.int8)
    matrix = codes.reshape(codes.shape[0], -1)
    singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    tolerance = (singular[0] if singular.size else 0.0) * max(matrix.shape) * np.finfo(np.float64).eps
    numerical_rank = int(np.sum(singular > tolerance))
    unique_rows = int(np.unique(matrix, axis=0).shape[0])
    return {
        "matrix_shape": list(matrix.shape),
        "numerical_rank": numerical_rank,
        "full_rank_bound": min(matrix.shape),
        "rank_fraction": numerical_rank / min(matrix.shape),
        "unique_rows": unique_rows,
        "duplicate_rows": int(matrix.shape[0] - unique_rows),
        "zero_rows": int(np.all(matrix == 0, axis=1).sum()),
        "zero_columns": int(np.all(matrix == 0, axis=0).sum()),
        "stable_rank": float(np.square(singular).sum() / np.square(singular[0])) if singular.size and singular[0] else 0.0,
        "top_singular_energy_fraction": float(np.square(singular[0]) / np.square(singular).sum()) if singular.size and np.square(singular).sum() else 0.0,
    }


def autopsy(output: Path) -> dict[str, Any]:
    records, canonical, legacy = _load_records()
    _, _, _, bits_module = _book_modules()
    runtime = _runtime_module()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = runtime.SemanticTokenRenderer(96).eval()
    model_state = model.state_dict()
    decoded_state = OrderedDict(
        (record.schema.name, torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)))
        for record in records
    )
    if tuple(decoded_state) != tuple(model_state):
        raise MZ2Error("fixed decoder schema order differs from the frozen renderer state order")
    if any(tuple(decoded_state[name].shape) != tuple(model_state[name].shape) for name in decoded_state):
        raise MZ2Error("fixed decoder schema shape differs from the frozen renderer")
    model.load_state_dict(decoded_state, strict=True)

    tensor_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    derived_names: list[str] = []
    tensor_root = output / "retained/autopsy/tensors"
    for index, record in enumerate(records):
        safe_name = record.schema.name.replace(".", "_")
        values = np.ascontiguousarray(record.values, dtype=np.float32)
        storage = _raw_record_payload(record, bits_module)
        payloads = {
            "decoded_values": atomic_npy(tensor_root / f"{index:02d}_{safe_name}.values.f32.npy", values),
            "stored_record": atomic_bytes(tensor_root / f"{index:02d}_{safe_name}.stored.bin", storage),
        }
        codes = None if record.codes is None else np.ascontiguousarray(record.codes, dtype=np.int8)
        if codes is not None:
            payloads["codes"] = atomic_npy(tensor_root / f"{index:02d}_{safe_name}.codes.i8.npy", codes)
        unique_values = np.unique(values)
        derivable = "none"
        if unique_values.size == 1 and float(unique_values[0]) == 0.0:
            derivable = "structural_zero"
        elif unique_values.size == 1 and float(unique_values[0]) == 1.0:
            derivable = "structural_one"
        if derivable != "none":
            derived_names.append(record.schema.name)
        tensor_rows.append(
            {
                "index": index,
                "name": record.schema.name,
                "shape": list(record.schema.shape),
                "format": record.format,
                "elements": int(values.size),
                "stored_bytes": len(storage),
                "stored_h0_bytes": h0_bytes(storage),
                "zero_fraction": float(np.mean(values == 0)),
                "unique_decoded_values": int(unique_values.size),
                "derive_at_decode": derivable,
                "matrix": _matrix_stats(record),
                "payloads": payloads,
            }
        )
        missing = OrderedDict((name, value) for name, value in decoded_state.items() if name != record.schema.name)
        refused = False
        message = ""
        try:
            model.load_state_dict(missing, strict=True)
        except RuntimeError as error:
            refused = True
            message = str(error).splitlines()[0]
        if not refused:
            raise MZ2Error(f"receiver accepted deletion of required tensor {record.schema.name}")
        deletion_rows.append(
            {
                "name": record.schema.name,
                "receiver_strict_load_refused": True,
                "module_mapping": record.schema.name.rsplit(".", 1)[0],
                "error_class": "missing state_dict key",
                "error_summary": message,
            }
        )

    exact_matrix_names = [row["name"] for row in tensor_rows if row["matrix"] is not None]
    exact_full_rank = [
        row["name"]
        for row in tensor_rows
        if row["matrix"] is not None and row["matrix"]["numerical_rank"] == row["matrix"]["full_rank_bound"]
    ]
    result = {
        "schema": "ddm_mz2_autopsy.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "source": {
            "f12_body": file_record(F12_BODY),
            "canonical_wans": {"bytes": len(canonical), "sha256": sha256_bytes(canonical)},
            "legacy_layout": {"bytes": len(legacy), "sha256": sha256_bytes(legacy)},
        },
        "tensor_denominator": len(records),
        "receiver_consumed_tensors": len(deletion_rows),
        "receiver_consumption_proof": "F12 fixed-schema decode plus actual strict=True load; deleting each one of 38 mapped keys is refused",
        "forward_topology": "all mapped leaf modules are traversed by SemanticTokenRenderer.forward: embeddings, coord_mix, four complete blocks, and head",
        "deletion_census": deletion_rows,
        "tensors": tensor_rows,
        "derive_at_decode": {
            "tested_tensors": len(records),
            "identity_derivable_count": len(derived_names),
            "identity_derivable_names": derived_names,
            "allowed_constants": [0.0, 1.0],
            "verdict": "no video-derived constant is treated as free receiver code",
        },
        "rank_census": {
            "quantized_matrix_denominator": len(exact_matrix_names),
            "numerically_full_rank_count": len(exact_full_rank),
            "numerically_full_rank_names": exact_full_rank,
            "scope": "decoded int4 code matrices flattened by output row; numerical SVD is diagnostic, not an exact factorization proof",
        },
        "all_materialized_tensor_payloads_retained": True,
        "live_e960_touched": False,
    }
    atomic_json(output / "AUTOPSY_RESULT.json", result)
    return result


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    if bits < 1 or bits > 16 or np.any(flat >= (1 << bits)):
        raise MZ2Error("unsigned bit-pack range violation")
    positions = np.arange(bits, dtype=np.uint32)
    bitstream = ((flat[:, None] >> positions[None]) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitstream, bitorder="little").tobytes()


def _unpack_unsigned(blob: memoryview, count: int, bits: int) -> tuple[np.ndarray, memoryview]:
    size = (count * bits + 7) // 8
    if len(blob) < size:
        raise MZ2Error("truncated unsigned bit stream")
    packed = np.frombuffer(blob[:size], dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
    values = (stream.astype(np.uint32) * (1 << np.arange(bits, dtype=np.uint32))).sum(axis=1)
    return values, blob[size:]


def _code_payloads(record, bits_module) -> dict[int, bytes]:
    codes = np.asarray(record.codes, dtype=np.int8)
    flat = codes.reshape(-1)
    dense = bits_module.pack_signed(flat, 4)
    mask = flat != 0
    sparse = np.packbits(mask, bitorder="little").tobytes() + bits_module.pack_signed(flat[mask], 4)
    matrix = codes.reshape(codes.shape[0], -1)
    unique, inverse = np.unique(matrix, axis=0, return_inverse=True)
    index_bits = max(1, math.ceil(math.log2(max(1, len(unique)))))
    rowdict = struct.pack("<H", len(unique)) + _pack_unsigned(inverse, index_bits) + bits_module.pack_signed(unique.reshape(-1), 4)
    return {MODE_DENSE: dense, MODE_SPARSE: sparse, MODE_ROWDICT: rowdict}


def pack_exact(records: Sequence[Any], strategy: str) -> tuple[bytes, list[dict[str, Any]]]:
    _, _, _, bits_module = _book_modules()
    qrecords = [record for record in records if not record.schema.is_fp16]
    modes: list[int] = []
    payloads: list[bytes] = []
    rows: list[dict[str, Any]] = []
    for record in qrecords:
        options = _code_payloads(record, bits_module)
        if strategy == "dense":
            mode = MODE_DENSE
        elif strategy == "hybrid_sparse":
            mode = min((MODE_DENSE, MODE_SPARSE), key=lambda item: (len(options[item]), item))
        elif strategy == "hybrid_rowdict":
            mode = min((MODE_DENSE, MODE_ROWDICT), key=lambda item: (len(options[item]), item))
        elif strategy == "hybrid_all":
            mode = min(options, key=lambda item: (len(options[item]), item))
        else:
            raise MZ2Error(f"unknown exact representation strategy: {strategy}")
        modes.append(mode)
        payloads.append(options[mode])
        rows.append(
            {
                "name": record.schema.name,
                "selected_mode": MODE_NAMES[mode],
                "selected_code_bytes": len(options[mode]),
                "dense_code_bytes": len(options[MODE_DENSE]),
                "sparse_code_bytes": len(options[MODE_SPARSE]),
                "row_dictionary_code_bytes": len(options[MODE_ROWDICT]),
            }
        )
    mode_bits = sum(mode << (2 * index) for index, mode in enumerate(modes)).to_bytes((2 * len(modes) + 7) // 8, "little")
    output = bytearray(MZ2E_MAGIC + bytes((MZ2E_VERSION, len(modes))) + mode_bits)
    qoffset = 0
    for record in records:
        if record.schema.is_fp16:
            output.extend(record.raw_fp16 or np.asarray(record.values, dtype="<f2").tobytes())
        else:
            output.extend(record.raw_scales or np.asarray(record.scales, dtype="<f2").tobytes())
            output.extend(payloads[qoffset])
            qoffset += 1
    return bytes(output), rows


def unpack_exact(blob: bytes):
    baseline, _, _, bits_module = _book_modules()
    if len(blob) < 6 or blob[:4] != MZ2E_MAGIC or blob[4] != MZ2E_VERSION:
        raise MZ2Error("invalid MZ2E header")
    qcount = int(blob[5])
    expected_qcount = sum(not schema.is_fp16 for schema in baseline.SEMANTIC_SCHEMA)
    if qcount != expected_qcount:
        raise MZ2Error("MZ2E quantized-record count differs")
    mode_bytes = (2 * qcount + 7) // 8
    mode_word = int.from_bytes(blob[6 : 6 + mode_bytes], "little")
    modes = [(mode_word >> (2 * index)) & 3 for index in range(qcount)]
    if any(mode not in MODE_NAMES for mode in modes):
        raise MZ2Error("MZ2E reserved representation mode")
    remaining = memoryview(blob)[6 + mode_bytes :]
    records = []
    qoffset = 0
    for schema in baseline.SEMANTIC_SCHEMA:
        if schema.is_fp16:
            size = schema.count * 2
            if len(remaining) < size:
                raise MZ2Error(f"truncated MZ2E fp16 tensor {schema.name}")
            raw = bytes(remaining[:size])
            remaining = remaining[size:]
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(schema.shape)
            records.append(baseline.TensorStorage(schema, "fp16", values, None, None, raw_fp16=raw))
            continue
        scale_size = schema.scale_count * 2
        if len(remaining) < scale_size:
            raise MZ2Error(f"truncated MZ2E scales {schema.name}")
        raw_scales = bytes(remaining[:scale_size])
        remaining = remaining[scale_size:]
        mode = modes[qoffset]
        if mode == MODE_DENSE:
            size = (schema.count + 1) // 2
            raw_codes = bytes(remaining[:size])
            remaining = remaining[size:]
            codes = np.asarray(bits_module.unpack_signed(raw_codes, schema.count, 4), dtype=np.int8)
        elif mode == MODE_SPARSE:
            mask_size = (schema.count + 7) // 8
            if len(remaining) < mask_size:
                raise MZ2Error(f"truncated MZ2E sparse mask {schema.name}")
            mask = np.unpackbits(np.frombuffer(remaining[:mask_size], dtype=np.uint8), bitorder="little")[: schema.count].astype(bool)
            remaining = remaining[mask_size:]
            count = int(mask.sum())
            size = (count + 1) // 2
            nonzero = np.asarray(bits_module.unpack_signed(bytes(remaining[:size]), count, 4), dtype=np.int8)
            remaining = remaining[size:]
            codes = np.zeros(schema.count, dtype=np.int8)
            codes[mask] = nonzero
        else:
            if len(remaining) < 2:
                raise MZ2Error(f"truncated MZ2E dictionary header {schema.name}")
            unique_count = struct.unpack_from("<H", remaining)[0]
            remaining = remaining[2:]
            if unique_count < 1:
                raise MZ2Error(f"empty MZ2E dictionary {schema.name}")
            rows = schema.shape[0]
            columns = schema.count // rows
            index_bits = max(1, math.ceil(math.log2(unique_count)))
            inverse, remaining = _unpack_unsigned(remaining, rows, index_bits)
            size = (unique_count * columns + 1) // 2
            unique = np.asarray(bits_module.unpack_signed(bytes(remaining[:size]), unique_count * columns, 4), dtype=np.int8).reshape(unique_count, columns)
            remaining = remaining[size:]
            if np.any(inverse >= unique_count):
                raise MZ2Error(f"MZ2E dictionary index overflow {schema.name}")
            codes = unique[inverse].reshape(-1)
        codes = codes.reshape(schema.shape)
        if np.any(codes == -8):
            raise MZ2Error(f"MZ2E reserved int4 code {schema.name}")
        scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32)
        scale_shape = [1] * len(schema.shape)
        scale_shape[-1 if schema.name.endswith("embed.weight") else 0] = schema.scale_count
        values = codes.astype(np.float32) * scales.reshape(scale_shape)
        records.append(baseline.TensorStorage(schema, "w4", values, scales, codes, raw_scales=raw_scales))
        qoffset += 1
    if remaining:
        raise MZ2Error(f"MZ2E has {len(remaining)} trailing bytes")
    return tuple(records)


def _same_state(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return len(left) == len(right) and all(
        a.schema == b.schema and np.array_equal(a.values, b.values) for a, b in zip(left, right, strict=True)
    )


def _retain_archive_candidate(output: Path, candidate_id: str, semantic_raw: bytes, parser: str) -> dict[str, Any]:
    hpac_stream, _, carrier_stream = _source_streams()
    semantic_stream = mz1._brotli(semantic_raw, 11)
    model = rx1.pack_rx1_model(
        hpac_stream,
        semantic_stream,
        carrier_stream,
        codec_id=rx1.RX1_CODEC_BROTLI,
        table_mode=rx1.RX1_TABLE_ON,
    )
    member = model + mz1.SOURCE_RESIDUAL.read_bytes() + mz1.SOURCE_TOKEN.read_bytes()
    archive = rx1.deterministic_zip(member)
    root = output / "retained/candidates" / candidate_id
    payloads = {
        "semantic_raw": atomic_bytes(root / f"semantic.{parser}.bin", semantic_raw),
        "semantic_brotli_q11": atomic_bytes(root / "semantic.br", semantic_stream),
        "model": atomic_bytes(root / "models.rx1m", model),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", rx1.deterministic_zip(member)),
    }
    if payloads["archive"]["sha256"] != payloads["archive_repeat"]["sha256"]:
        raise MZ2Error(f"nondeterministic archive build: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "parser": parser,
        "payloads": payloads,
        "archive_bytes": len(archive),
        "delta_archive_bytes_vs_current": len(archive) - CURRENT_ARCHIVE_BYTES,
        "semantic_raw_bytes": len(semantic_raw),
        "semantic_stream_bytes": len(semantic_stream),
        "delta_semantic_stream_bytes_vs_current": len(semantic_stream) - EXPECTED_SEMANTIC_STREAM_BYTES,
        "projected_rate_only_delta_score": (len(archive) - CURRENT_ARCHIVE_BYTES) * 25 / RATE_DENOMINATOR,
        "repeat_byte_identical": True,
        "shipping_receiver_status": "adapter_required_if_selected",
        "score_claim": False,
    }


def exact_race(output: Path) -> dict[str, Any]:
    records, _, _ = _load_records()
    candidates = []
    for strategy in ("dense", "hybrid_sparse", "hybrid_rowdict", "hybrid_all"):
        payload, per_tensor = pack_exact(records, strategy)
        decoded = unpack_exact(payload)
        if not _same_state(records, decoded):
            raise MZ2Error(f"exact representation parse-back differs: {strategy}")
        candidate = _retain_archive_candidate(output, f"exact_{strategy}", payload, "mz2e")
        candidate["decoded_state_identity"] = True
        candidate["selection"] = strategy
        candidate["per_tensor_modes"] = per_tensor
        candidates.append(candidate)
    control = {
        "candidate_id": "current_f12_control",
        "archive_bytes": CURRENT_ARCHIVE_BYTES,
        "delta_archive_bytes_vs_current": 0,
        "semantic_raw_bytes": EXPECTED_F12[0],
        "semantic_stream_bytes": EXPECTED_SEMANTIC_STREAM_BYTES,
        "delta_semantic_stream_bytes_vs_current": 0,
        "decoded_state_identity": True,
        "shipping_receiver_status": "current receiver closed",
        "archive": require_file(mz1.SOURCE_ARCHIVE, mz1.EXPECTED["archive"]),
    }
    winner = min([control, *candidates], key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    result = {
        "schema": "ddm_mz2_exact_race.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "candidate_denominator": len(candidates),
        "control": control,
        "candidates": candidates,
        "winner": winner,
        "required_savings_for_sub015_at_fixed_distortion": REQUIRED_SAVINGS,
        "winner_meets_required_savings": -winner["delta_archive_bytes_vs_current"] >= REQUIRED_SAVINGS,
        "verdict_scope": "INSTANCE: exact e480b decoded semantic state; fixed-schema dense, zero-sparse, and row-dictionary forms under RX1 split Brotli q11",
        "all_candidate_payloads_retained": True,
    }
    atomic_json(output / "EXACT_RESULT.json", result)
    return result


def _selected_mixed(records: Sequence[Any]) -> tuple[bytes, OrderedDict[str, torch.Tensor]]:
    state = OrderedDict(
        (record.schema.name, torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)))
        for record in records
    )
    allocation = OrderedDict(
        (name, 3 if name in sm3.SELECTED_MIXED_Q3_NAMES else 4)
        for name in sd1.quantized_names(state)
    )
    return sd1.pack_semantic_state(state, allocation, legacy_int4=False)


def _state_delta(
    restored: Mapping[str, torch.Tensor],
    template: Mapping[str, torch.Tensor],
    allocation: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows = []
    total_changed = 0
    total_elements = 0
    for name in restored:
        difference = restored[name] - template[name]
        changed = int(torch.count_nonzero(difference).item())
        elements = int(difference.numel())
        total_changed += changed
        total_elements += elements
        rows.append(
            {
                "name": name,
                "bits": None if allocation is None else int(allocation.get(name, 16)),
                "changed_elements": changed,
                "elements": elements,
                "changed_fraction": changed / elements,
                "mse": float(torch.mean(difference.square()).item()),
                "max_abs": float(torch.max(torch.abs(difference)).item()),
            }
        )
    return {
        "changed_elements": total_changed,
        "elements": total_elements,
        "changed_fraction": total_changed / total_elements,
        "per_tensor": rows,
    }


def score_gate(output: Path) -> dict[str, Any]:
    records, _, _ = _load_records()
    payload, expected_state = _selected_mixed(records)
    template = OrderedDict(
        (record.schema.name, torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)))
        for record in records
    )
    restored, allocation, format_name = sd1.unpack_semantic_state(payload, template)
    if tuple(restored) != tuple(expected_state) or any(not torch.equal(restored[name], expected_state[name]) for name in restored):
        raise MZ2Error("current-state selected mixed-bit parse-back differs")
    candidate = _retain_archive_candidate(output, "score_gated_selected_mixed_q3q4", payload, "sd1m")
    candidate.update(
        {
            "measurement_decoder_identity": True,
            "format_name": format_name,
            "q3_tensor_names": sorted(name for name, bits in allocation.items() if bits == 3),
            "q4_tensor_names": sorted(name for name, bits in allocation.items() if bits == 4),
            "prior_evidence": {
                "vehicle": "PR130 ancestor, not transferable as a current score",
                "axis": "[macOS-CPU advisory; n600; pose unmeasured]",
                "archive_savings_bytes": 848,
                "semantic_leg_delta_score": -0.000423928448671,
            },
            "current_n600_score": None,
            "decoded_state_delta": _state_delta(restored, template, allocation),
            "admission_status": "NOT_ADMITTED",
            "admission_reason": "charter did not claim the one-full-n600 scorer slot; current-state distortion and pose are unmeasured",
        }
    )
    structured_candidates = []
    for keep_percent in (25, 37, 50, 62, 75, 87):
        prune_payload, prune_expected, metadata = sm3.pack_prune_candidate(template, keep_percent)
        prune_restored = sm3.unpack_prune_candidate(prune_payload, template)
        if tuple(prune_restored) != tuple(prune_expected) or any(
            not torch.equal(prune_restored[name], prune_expected[name]) for name in prune_restored
        ):
            raise MZ2Error(f"current-state structured-prune parse-back differs at keep={keep_percent}")
        prune = _retain_archive_candidate(
            output,
            f"score_gated_film_row_prune_keep{keep_percent}",
            prune_payload,
            "sm3r",
        )
        prune.update(
            {
                "measurement_decoder_identity": True,
                "keep_percent": keep_percent,
                "metadata": metadata,
                "decoded_state_delta": _state_delta(prune_restored, template),
                "current_n600_score": None,
                "admission_status": "NOT_ADMITTED",
                "admission_reason": "current Seg/Pose effect is unmeasured; the six-cell per-lever curve is retained for sequential n600 gating",
            }
        )
        structured_candidates.append(prune)
    byte_gate = candidate["archive_bytes"] < CURRENT_ARCHIVE_BYTES
    disposition = "QUEUED-WITH-A-FIRE-ORDER" if byte_gate else "FOLDED"
    queue = {
        "schema": "ddm_mz2_scorer_queue.v1",
        "disposition": disposition,
        "owner": "MAIN",
        "consumer_store": str((output / "SCORE_GATE_RESULT.json").resolve()),
        "candidate": candidate["payloads"]["archive"],
        "current_fire": False,
        "fire_trigger": (
            "No full n600 scorer job is active; a shipping SD1M receiver adapter parse-backs this exact archive; "
            "then run local advisory n600 on the exact retained archive and admit only if projected net delta score is below zero by more than 3.5e-6."
        ),
        "reason": "candidate clears the current archive-byte gate but has no current-vehicle Seg/Pose measurement" if byte_gate else "candidate does not clear the current archive-byte gate",
    }
    atomic_json(output / "SCORER_QUEUE.json", queue)
    structured_queue = {
        "schema": "ddm_mz2_structured_sparsity_queue.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str((output / "SCORE_GATE_RESULT.json").resolve()),
        "candidates": [row["payloads"]["archive"] for row in structured_candidates],
        "current_fire": False,
        "fire_trigger": (
            "The mixed-q3/q4 row has a current-vehicle verdict and no full n600 scorer job is active; "
            "install one strict SM3R receiver adapter, then evaluate the six retained keep-percent cells sequentially and stop when the Pareto direction is dominated."
        ),
        "reason": "all six current-state payloads are byte-closed but none has current Seg/Pose evidence",
    }
    atomic_json(output / "STRUCTURED_SPARSITY_QUEUE.json", structured_queue)
    distill_queue = {
        "schema": "ddm_mz2_distill_queue.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str((output / "SCORE_GATE_RESULT.json").resolve()),
        "current_fire": False,
        "fire_trigger": (
            "The live e960 burn is complete and no scorer/trainer occupies the governed slot; launch a deterministic, resume-from-disk width sweep with distinct per-stage checkpoints, "
            "distill from the exact current renderer, retain every learned payload, and score only receiver-closed archives."
        ),
        "reason": "a trained smaller renderer cannot be honestly materialized scorer-free during the untouchable live e960 burn",
        "required_mechanism": "distilled smaller semantic renderer; no proxy-only or untrained width truncation is admissible",
    }
    atomic_json(output / "DISTILL_QUEUE.json", distill_queue)
    result = {
        "schema": "ddm_mz2_score_gate.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "candidate": candidate,
        "structured_sparsity_candidate_denominator": len(structured_candidates),
        "structured_sparsity_candidates": structured_candidates,
        "byte_gate_pass": byte_gate,
        "score_gate_pass": False,
        "scorer_queue": queue,
        "structured_sparsity_queue": structured_queue,
        "distill_queue": distill_queue,
        "noise_floor": 3.5e-6,
        "verdict_scope": "INSTANCE: current e480b semantic state re-quantized by the prior SD1 selected allocation; byte gate only, no current distortion verdict",
        "all_candidate_payloads_retained": True,
    }
    atomic_json(output / "SCORE_GATE_RESULT.json", result)
    return result


def _retention_inventory(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_RESULT.json", "RETENTION_INVENTORY.json"}
    files = [path for path in output.rglob("*") if path.is_file() and path.name not in excluded]
    rows = [file_record(path) for path in sorted(files)]
    result = {
        "schema": "ddm_mz2_retention_inventory.v1",
        "complete": True,
        "file_denominator": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "excluded_to_avoid_recursive_hashing": sorted(excluded),
        "files": rows,
    }
    atomic_json(output / "RETENTION_INVENTORY.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    autopsy_result = json.loads((output / "AUTOPSY_RESULT.json").read_text(encoding="utf-8"))
    exact_result = json.loads((output / "EXACT_RESULT.json").read_text(encoding="utf-8"))
    score_result = json.loads((output / "SCORE_GATE_RESULT.json").read_text(encoding="utf-8"))
    exact_winner = exact_result["winner"]
    exact_savings = -int(exact_winner["delta_archive_bytes_vs_current"])
    score_candidate = score_result["candidate"]
    t4 = {
        "schema": "ddm_mz2_t4_fire_order.v1",
        "disposition": "FOLDED",
        "owner": "MAIN",
        "consumer_store": str((output / "FINAL_RESULT.json").resolve()),
        "current_fire": False,
        "fire_trigger": (
            "A retained candidate beats 183502 bytes, ships a strict receiver, passes current-vehicle local advisory n600 "
            "by more than the 3.5e-6 noise floor, and has a sealed exact archive SHA-256; dispatch to T4, never Modal."
        ),
        "reason": "no byte-distinct candidate is both receiver-closed and current-vehicle score-admitted",
    }
    atomic_json(output / "T4_FIRE_ORDER.json", t4)
    carrier_queue = {
        "schema": "ddm_mz2_carrier_representation_queue.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str((output / "FINAL_RESULT.json").resolve()),
        "current_fire": False,
        "fire_trigger": (
            "The semantic mixed-bit, structured-sparsity, and distilled-width queues have terminal verdicts but leave the frontier above 0.15; "
            "then attack the exact 22219-byte CAP1 decoded state with a representation not already covered by PK2, PK4, PS135B, or MZ1."
        ),
        "reason": "the charter permits semantic-first scope; carrier is second and must preserve the same decoded values rather than reopen pose solving",
        "prior_art_required": ["PK2", "PK4", "PS135B", "MZ1", "FD135"],
    }
    atomic_json(output / "CARRIER_QUEUE.json", carrier_queue)
    result = {
        "schema": "ddm_mz2_final.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "measured": {
            "receiver_consumption": f"{autopsy_result['receiver_consumed_tensors']}/{autopsy_result['tensor_denominator']} tensors",
            "identity_exact_representation_candidates": exact_result["candidate_denominator"],
            "exact_winner": exact_winner,
            "score_gated_candidate_archive_bytes": score_candidate["archive_bytes"],
            "score_gated_candidate_delta_bytes": score_candidate["delta_archive_bytes_vs_current"],
        },
        "not_measured": [
            "no current-vehicle SegNet or PoseNet score for the mixed-bit candidate",
            "no contest CPU/CUDA evaluation",
            "no shipping-receiver decode of any byte-distinct candidate",
        ],
        "boundaries": {
            "lossless_same-state_representation": "closed in tested fixed-schema dense/sparse/row-dictionary scope" if exact_savings <= 0 else "open: exact candidate is smaller but needs a shipping adapter",
            "derive_at_decode": "closed for structural zero/one constants on all receiver-consumed tensors",
            "lowrank_vq": "consumed SM3/SM4/SV3 negative evidence; not rerun",
            "mixed_q3q4": "byte-gated current-state candidate retained; current distortion verdict queued",
        },
        "t4_fire_order": t4,
        "scorer_queue": score_result["scorer_queue"],
        "structured_sparsity_queue": score_result["structured_sparsity_queue"],
        "distill_queue": score_result["distill_queue"],
        "carrier_queue": carrier_queue,
        "all_materialized_payloads_retained": True,
        "required_savings_for_sub015": REQUIRED_SAVINGS,
        "own_vehicle_frontier": {
            "score": CURRENT_SCORE,
            "archive_bytes": CURRENT_ARCHIVE_BYTES,
            "archive_sha256": CURRENT_ARCHIVE_SHA256,
            "axis": "[contest-CUDA T4, n600]",
            "pointer_moved": False,
        },
        "verdict_scope": "current e480b frozen semantic section only; scorer-free except inherited, explicitly non-transferable evidence",
        "live_e960_touched": False,
    }
    atomic_json(output / "FINAL_RESULT.json", result)
    inventory = _retention_inventory(output)
    result["retention_inventory"] = file_record(output / "RETENTION_INVENTORY.json")
    result["retained_file_denominator"] = inventory["file_denominator"]
    atomic_json(output / "FINAL_RESULT.json", result)
    return result


def run_stage(output: Path, stage: str) -> dict[str, Any]:
    stages = [
        ("preflight", "PREFLIGHT.json", preflight),
        ("autopsy", "AUTOPSY_RESULT.json", autopsy),
        ("exact", "EXACT_RESULT.json", exact_race),
        ("score-gate", "SCORE_GATE_RESULT.json", score_gate),
        ("finalize", "FINAL_RESULT.json", finalize),
    ]
    selected = stages if stage == "all" else [item for item in stages if item[0] == stage]
    if not selected:
        raise MZ2Error(f"unknown stage: {stage}")
    result: dict[str, Any] = {}
    for name, receipt, function in selected:
        path = output / receipt
        if stage == "all" and path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("complete") is not True:
                raise MZ2Error(f"resume checkpoint is incomplete: {path}")
            continue
        if name != "preflight" and not (output / "PREFLIGHT.json").is_file():
            raise MZ2Error("preflight checkpoint is required before later stages")
        result = function(output)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "autopsy", "exact", "score-gate", "finalize", "all"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=None, help="must name OUTPUT; stage receipts are the byte-close resume checkpoints")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if args.resume_from is not None and args.resume_from.resolve() != output:
        raise MZ2Error("--resume-from must equal --output for this staged runner")
    result = run_stage(output, args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
