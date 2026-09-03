#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""QX3 scorer-free receiver closure for the QX2 conditioning baseline.

The decoder reconstructs the QBT model and latents carried by QX1, evaluates
that model on all 600 pair identifiers, and then applies an optional counted
correction before decoding QX2's address-free event stream.  The correction
and QX2 event stream are compressed jointly as one QXE section, so the exact
24,093-byte section gate is measured without charging a second section header.

This is an exact receiver/rate experiment.  It loads no scorer and makes no
contest-score or distortion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from collections import Counter
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from tac.optimization.s2_partition_seed import PartitionEvent, decode_partition_seed

STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx3")
QX1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qx1")
QX2_STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx2")
QX1_CORE: Final = QX1_STORE / "retained/envelopes/core_without_events_exceptions/envelope.qxe"
QX1_CENSUS: Final = QX1_STORE / "SECTION_CENSUS.json"
QX2_RESULT: Final = QX2_STORE / "RESULT.json"
QX2_BASELINE: Final = QX2_STORE / "retained/baseline/c1_baseline_labels.u8"
QX2_RUNNER: Final = REPO / "experiments/ddm_qx2_events_section_redesign.py"
EVENT_SOURCE: Final = QX1_STORE / "retained/sections/08_events_exceptions_explicit_address_control/raw.bin"
GT_CACHE: Final = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
QBT_CONTAINER: Final = Path(
    "/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/final_reencode/reencode_payloads.tar"
)
QBZ1_RESULT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration/FIT_RESULT.json"
)

PINS: Final = {
    QX1_CORE: "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95",
    QX1_CENSUS: "d552955b1e7be08f03c77e3508756ad3e1dead9be759bf562f3cfc9ec8296db6",
    QX2_RESULT: "b3a63070260ca4d8d6ea23ec7395bb3156b2cbdae91c1a27bca2e0d82b63e234",
    QX2_BASELINE: "02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34",
    QX2_RUNNER: "88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c",
    EVENT_SOURCE: "df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc",
    GT_CACHE: "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    QBT_CONTAINER: "4c16e6c045768b2dee62f59ac9a2a27b7386280dfccff3dd5331a8d9509d95f7",
    QBZ1_RESULT: "69b33e5d393deff7f1fcd76844cf524d7c19691f431aa399a876b2ad1ce227bf",
}

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
SITES: Final = N_PAIRS * HEIGHT * WIDTH
EVENTS: Final = 17_926
SECTION_CAP_BYTES: Final = 24_093
ARCHIVE_GATE_EXCLUSIVE: Final = 137_986
QX1_CORE_ARCHIVE_BYTES: Final = 113_844
QX2_EVENT_CODED_BYTES: Final = 22_661
QX2_TARGET_SHA256: Final = "36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68"
QBT_MODEL_RAW_SHA256: Final = "2280c2d3c54d1781559ec130123a05ec664dbdf347b04f379805bfbe67f59085"
MINIMUM_FREE_BYTES: Final = 1_000_000_000
AXIS: Final = "[scorer-free exact receiver/rate measurement]"

QXE_HEADER: Final = struct.Struct(">4sBBH")
QXE_SECTION: Final = struct.Struct(">BBHII32sI")
QXT_HEADER: Final = struct.Struct(">4sBBB")
QBT_TENSOR_HEADER: Final = struct.Struct(">BBBBfII")
DENSE_HEADER: Final = struct.Struct(">4sBHHH32s32s")
SPARSE_HEADER: Final = struct.Struct(">4sBHHHQ32s32s")
CLOSURE_HEADER: Final = struct.Struct(">4sBBII32s32s")
CODECS: Final = {"brotli_q11": 1, "lzma9e": 2, "zlib9": 3}
CODEC_NAMES: Final = {value: key for key, value in CODECS.items()}
FORM_IDS: Final = {"dense_delta": 1, "sparse_u32_delta": 2}
FORM_NAMES: Final = {value: key for key, value in FORM_IDS.items()}


class QX3Error(RuntimeError):
    """An exactness, custody, or receiver gate failed closed."""


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def require_fact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise QX3Error(f"{label} is absent: {path}")
    observed = fact(path)
    if observed["sha256"] != expected_sha256:
        raise QX3Error(f"{label} SHA drifted: {observed['sha256']} != {expected_sha256}")
    return observed


def compress(codec: str, raw: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if codec == "lzma9e":
        return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    raise QX3Error(f"unknown codec: {codec}")


def decompress(codec: str, coded: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.decompress(coded)
    if codec == "lzma9e":
        return lzma.decompress(coded)
    if codec == "zlib9":
        return zlib.decompress(coded)
    raise QX3Error(f"unknown codec: {codec}")


def parse_qxe(packet: bytes, expected_count: int) -> tuple[list[bytes], dict[int, bytes], dict[int, str]]:
    if len(packet) < QXE_HEADER.size:
        raise QX3Error("QXE packet is truncated")
    magic, version, flags, count = QXE_HEADER.unpack_from(packet)
    if (magic, version, flags, count) != (b"QXE1", 1, 0, expected_count):
        raise QX3Error("QXE packet identity drifted")
    records: list[bytes] = []
    raws: dict[int, bytes] = {}
    codecs: dict[int, str] = {}
    offset = QXE_HEADER.size
    for expected_id in range(1, expected_count + 1):
        start = offset
        if offset + QXE_SECTION.size > len(packet):
            raise QX3Error("QXE section header is truncated")
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, crc = QXE_SECTION.unpack_from(packet, offset)
        offset += QXE_SECTION.size
        end = offset + coded_len
        if section_id != expected_id or codec_id not in CODEC_NAMES or reserved or end > len(packet):
            raise QX3Error("QXE section envelope drifted")
        coded = packet[offset:end]
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if (
            len(raw) != raw_len
            or sha256_bytes(raw) != raw_sha.hex()
            or zlib.crc32(coded) & 0xFFFFFFFF != crc
        ):
            raise QX3Error("QXE section integrity failed")
        records.append(packet[start:end])
        raws[section_id] = raw
        codecs[section_id] = CODEC_NAMES[codec_id]
        offset = end
    if offset != len(packet):
        raise QX3Error("QXE packet has trailing bytes")
    return records, raws, codecs


def reassemble_qbt_model(
    groups: Sequence[bytes], *, expected_sha256: str = QBT_MODEL_RAW_SHA256
) -> tuple[bytes, dict[str, Any]]:
    expected = ((1, 28), (2, 10), (3, 4))
    records: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for raw, (group_id, tensor_count) in zip(groups, expected, strict=True):
        if len(raw) < QXT_HEADER.size:
            raise QX3Error("QXT model group is truncated")
        magic, observed_group, reserved, observed_count = QXT_HEADER.unpack_from(raw)
        if (magic, observed_group, reserved, observed_count) != (b"QXT1", group_id, 0, tensor_count):
            raise QX3Error("QXT model-group identity drifted")
        view = memoryview(raw)
        offset = QXT_HEADER.size
        names: list[str] = []
        for _ in range(tensor_count):
            start = offset
            if offset + QBT_TENSOR_HEADER.size > len(view):
                raise QX3Error("QXT tensor header is truncated")
            name_len, _bits, ndim, reserved, _scale, count, packed_len = QBT_TENSOR_HEADER.unpack_from(
                view, offset
            )
            offset += QBT_TENSOR_HEADER.size
            end_name = offset + name_len
            end_shape = end_name + 2 * ndim
            end_record = end_shape + packed_len
            if reserved or not name_len or not ndim or end_record > len(view):
                raise QX3Error("QXT tensor record is invalid")
            name = bytes(view[offset:end_name]).decode("ascii")
            shape = tuple(struct.unpack_from(">H", view, end_name + 2 * index)[0] for index in range(ndim))
            if math.prod(shape) != count or name in records:
                raise QX3Error("QXT tensor shape/count or unique-name identity drifted")
            records[name] = bytes(view[start:end_record])
            names.append(name)
            offset = end_record
        if offset != len(view):
            raise QX3Error("QXT model group has trailing bytes")
        rows.append(
            {
                "group_id": group_id,
                "tensor_count": tensor_count,
                "tensor_names": names,
                "raw_bytes": len(raw),
                "raw_sha256": sha256_bytes(raw),
            }
        )
    model_raw = (
        b"QBT1"
        + struct.pack(">H", sum(count for _, count in expected))
        + b"".join(records[name] for name in sorted(records))
    )
    if sha256_bytes(model_raw) != expected_sha256:
        raise QX3Error("reassembled QBT model is not the pinned ancestor model")
    params = qbf1.decode_model(model_raw)
    if set(params) != set(qbf1.expected_param_shapes()):
        raise QX3Error("reassembled QBT tensor roster drifted")
    return model_raw, {"groups": rows, "model_raw_bytes": len(model_raw), "model_raw_sha256": sha256_bytes(model_raw)}


def model_from_core(packet: bytes) -> tuple[qbt1.QBFLOWTorch, dict[str, Any]]:
    _records, sections, codecs = parse_qxe(packet, 7)
    model_raw, reassembly = reassemble_qbt_model((sections[2], sections[3], sections[4]))
    params = qbf1.decode_model(model_raw)
    meta = qbf1.decode_latent_meta(sections[5])
    latent_records = qbf1.decode_latent_table(sections[6])
    if set(latent_records) != set(range(N_PAIRS)):
        raise QX3Error("QX1 joint state does not carry all 600 QBT latent records")
    boundary = np.stack(
        [
            qbf1.dequantize(
                latent_records[pair][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)
            )
            for pair in range(N_PAIRS)
        ]
    )
    interior = np.stack(
        [
            qbf1.dequantize(
                latent_records[pair][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)
            )
            for pair in range(N_PAIRS)
        ]
    )
    model = qbt1.QBFLOWTorch(params, boundary, interior)
    model.eval()
    return model, {
        "section_codecs": {str(key): value for key, value in sorted(codecs.items())},
        "reassembly": reassembly,
        "config_sha256": sha256_bytes(sections[1]),
        "latent_meta_sha256": sha256_bytes(sections[5]),
        "joint_state_sha256": sha256_bytes(sections[6]),
        "pose_stream_sha256": sha256_bytes(sections[7]),
    }


def checkpoint_cursor(path: Path, schema: str) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise QX3Error(f"checkpoint schema drifted: {path}")
    return int(payload["cursor"])


def derive_decoder_baseline(store: Path, core: bytes) -> tuple[Path, dict[str, Any]]:
    output = store / "retained/derived/qx1_decoder_baseline.u8"
    stage_path = store / "checkpoints/STAGE1_DERIVED_BASELINE.json"
    if stage_path.is_file() and output.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("complete") and fact(output) == stage.get("derived_baseline"):
            return output, stage
    model, model_trace = model_from_core(core)
    partial = output.with_suffix(".u8.partial")
    cursor_path = store / "checkpoints/STAGE1_CURSOR.json"
    cursor = checkpoint_cursor(cursor_path, "ddm_qx3_stage1_cursor.v1")
    output.parent.mkdir(parents=True, exist_ok=True)
    if cursor == 0:
        with partial.open("wb") as handle:
            handle.truncate(SITES)
    if not partial.is_file() or partial.stat().st_size != SITES:
        raise QX3Error("derived-baseline partial file is absent or has wrong geometry")
    mapped = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        for pair in range(cursor, N_PAIRS):
            outputs = model(torch.tensor([pair], dtype=torch.long), height=HEIGHT, width=WIDTH)
            mapped[pair] = outputs["class_logits"][0].argmax(dim=-1).cpu().numpy().astype(np.uint8)
            if (pair + 1) % 10 == 0 or pair + 1 == N_PAIRS:
                mapped.flush()
                atomic_json(
                    cursor_path,
                    {
                        "schema": "ddm_qx3_stage1_cursor.v1",
                        "cursor": pair + 1,
                        "partial": {"path": str(partial), "bytes": partial.stat().st_size},
                    },
                )
    del mapped
    os.replace(partial, output)
    native_result = json.loads(QBZ1_RESULT.read_text(encoding="utf-8"))
    native_mismatches = 0
    native_sites = 0
    mapped = np.memmap(output, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    native_facts: list[dict[str, Any]] = []
    for row in native_result["native_capacity_measurement"]["retained_chunks"]:
        path = Path(row["path"])
        observed = require_fact(path, row["sha256"], "QBZ1 retained native-field chunk")
        native_facts.append(observed)
        with np.load(path, allow_pickle=False) as payload:
            pair_ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
            native = np.asarray(payload["native_argmax_u8"], dtype=np.uint8)
            native_mismatches += int(np.count_nonzero(mapped[pair_ids] != native))
            native_sites += int(native.size)
    del mapped
    if native_sites != SITES or native_mismatches:
        raise QX3Error("fresh QX1 decode differs from retained exact quantized-packet native field")
    stage = {
        "schema": "ddm_qx3_derived_baseline.v1",
        "complete": True,
        "axis": AXIS,
        "selection_mode": "full n600",
        "pairs": N_PAIRS,
        "sites": SITES,
        "derived_baseline": fact(output),
        "model_trace": model_trace,
        "retained_qbz1_native_field_chunks": native_facts,
        "fresh_decode_vs_retained_native_mismatches": native_mismatches,
        "receiver_inputs": ["QX1 QXE sections 1-6", "pair identifier", "generic QBT decoder code"],
        "encoder_only_inputs_used_by_receiver": [],
        "scorers_loaded": 0,
    }
    atomic_json(stage_path, stage)
    return output, stage


def baseline_frame(path: Path, pair: int) -> np.ndarray:
    return np.memmap(path, dtype=np.uint8, mode="r", offset=pair * HEIGHT * WIDTH, shape=(HEIGHT, WIDTH))


def characterize_mismatch(
    store: Path,
    derived_path: Path,
    target_path: Path,
    events: Sequence[PartitionEvent],
) -> dict[str, Any]:
    stage_path = store / "checkpoints/STAGE1_BIT_COMPARE.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text(encoding="utf-8"))
    derived = np.memmap(derived_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    target = np.memmap(target_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    transition_counts: Counter[str] = Counter()
    per_pair: list[int] = []
    first_mismatches: list[list[int]] = []
    contingency = np.zeros((5, 5), dtype=np.int64)
    for pair in range(N_PAIRS):
        source_frame = np.asarray(derived[pair])
        target_frame = np.asarray(target[pair])
        mismatch = source_frame != target_frame
        per_pair.append(int(np.count_nonzero(mismatch)))
        for source_class in range(5):
            source_mask = source_frame == source_class
            for target_class in range(5):
                contingency[source_class, target_class] += int(
                    np.count_nonzero(source_mask & (target_frame == target_class))
                )
        if len(first_mismatches) < 64 and np.any(mismatch):
            for row, col in np.argwhere(mismatch).tolist():
                first_mismatches.append(
                    [pair, row, col, int(source_frame[row, col]), int(target_frame[row, col])]
                )
                if len(first_mismatches) == 64:
                    break
    mismatch_count = int(sum(per_pair))
    for source_class in range(5):
        for target_class in range(5):
            if source_class != target_class and contingency[source_class, target_class]:
                transition_counts[f"{source_class}->{target_class}"] = int(
                    contingency[source_class, target_class]
                )
    event_site_mismatches = 0
    event_transition_counts: Counter[str] = Counter()
    for event in events:
        observed = int(derived[event.pair, event.row, event.col])
        expected = int(target[event.pair, event.row, event.col])
        if observed != expected:
            event_site_mismatches += 1
            event_transition_counts[f"{observed}->{expected}"] += 1
    ideal_bits = 0.0
    for source_class in range(5):
        row = contingency[source_class]
        ideal_bits += (
            math.lgamma(int(row.sum()) + 1)
            - sum(math.lgamma(int(value) + 1) for value in row)
        ) / math.log(2.0)
    del derived, target
    stage = {
        "schema": "ddm_qx3_bit_compare.v1",
        "complete": True,
        "axis": AXIS,
        "full_field": {
            "denominator_sites": SITES,
            "mismatches": mismatch_count,
            "mismatch_fraction": mismatch_count / SITES,
            "exact": mismatch_count == 0,
            "per_pair_min": min(per_pair),
            "per_pair_max": max(per_pair),
            "per_pair_mean": float(np.mean(per_pair)),
            "zero_mismatch_pairs": int(sum(value == 0 for value in per_pair)),
            "transition_counts": dict(sorted(transition_counts.items())),
            "first_mismatches": first_mismatches,
        },
        "event_sites": {
            "denominator_events": EVENTS,
            "mismatches": event_site_mismatches,
            "mismatch_fraction": event_site_mismatches / EVENTS,
            "exact": event_site_mismatches == 0,
            "transition_counts": dict(sorted(event_transition_counts.items())),
        },
        "conditional_multinomial_reference": {
            "bits": ideal_bits,
            "bytes_ceiling": math.ceil(ideal_bits / 8.0),
            "scope": "ideal enumerative assignment length conditional on the decoded source labels and observed 5-way row counts; not a universal Kolmogorov lower bound",
        },
        "contingency_source_rows_target_columns": contingency.tolist(),
    }
    atomic_json(stage_path, stage)
    return stage


def delta_codes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape:
        raise QX3Error("correction source/target geometry differs")
    codes = np.zeros(source.shape, dtype=np.uint8)
    mismatch = source != target
    ranks = target[mismatch] - (target[mismatch] > source[mismatch]).astype(np.uint8)
    codes[mismatch] = ranks + 1
    return codes


def apply_delta_codes(source: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if source.shape != codes.shape or np.any(codes > 4):
        raise QX3Error("correction code geometry or alphabet differs")
    target = source.copy()
    mismatch = codes != 0
    ranks = codes[mismatch] - 1
    target[mismatch] = ranks + (ranks >= source[mismatch]).astype(np.uint8)
    return target


def build_correction_forms(
    store: Path, derived_path: Path, target_path: Path, mismatch_count: int
) -> dict[str, Path]:
    stage_path = store / "checkpoints/STAGE2_CORRECTION_FORMS.json"
    dense_path = store / "retained/corrections/dense_delta/raw.bin"
    sparse_path = store / "retained/corrections/sparse_u32_delta/raw.bin"
    if stage_path.is_file() and dense_path.is_file() and sparse_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if fact(dense_path) == stage["forms"]["dense_delta"] and fact(sparse_path) == stage["forms"]["sparse_u32_delta"]:
            return {"dense_delta": dense_path, "sparse_u32_delta": sparse_path}
    source_sha = sha256_file(derived_path)
    target_sha = sha256_file(target_path)
    derived = np.memmap(derived_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    target = np.memmap(target_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    dense_path.parent.mkdir(parents=True, exist_ok=True)
    dense_partial = dense_path.with_suffix(".bin.partial")
    with dense_partial.open("wb") as handle:
        handle.write(
            DENSE_HEADER.pack(
                b"QXD1", 1, N_PAIRS, HEIGHT, WIDTH, bytes.fromhex(source_sha), bytes.fromhex(target_sha)
            )
        )
        for pair in range(N_PAIRS):
            handle.write(delta_codes(np.asarray(derived[pair]), np.asarray(target[pair])).tobytes())
            if (pair + 1) % 30 == 0:
                handle.flush()
                os.fsync(handle.fileno())
                atomic_json(
                    store / "checkpoints/STAGE2_DENSE_CURSOR.json",
                    {
                        "schema": "ddm_qx3_dense_cursor.v1",
                        "cursor": pair + 1,
                        "partial": {"path": str(dense_partial), "bytes": dense_partial.stat().st_size},
                    },
                )
    os.replace(dense_partial, dense_path)
    sparse_path.parent.mkdir(parents=True, exist_ok=True)
    sparse_partial = sparse_path.with_suffix(".bin.partial")
    observed_mismatches = 0
    record_dtype = np.dtype([("index", ">u4"), ("code", "u1")])
    with sparse_partial.open("wb") as handle:
        handle.write(
            SPARSE_HEADER.pack(
                b"QXS1",
                1,
                N_PAIRS,
                HEIGHT,
                WIDTH,
                mismatch_count,
                bytes.fromhex(source_sha),
                bytes.fromhex(target_sha),
            )
        )
        for pair in range(N_PAIRS):
            codes = delta_codes(np.asarray(derived[pair]), np.asarray(target[pair])).reshape(-1)
            positions = np.flatnonzero(codes).astype(np.uint64)
            records = np.empty(positions.size, dtype=record_dtype)
            records["index"] = positions + pair * HEIGHT * WIDTH
            records["code"] = codes[positions]
            handle.write(records.tobytes())
            observed_mismatches += int(positions.size)
            if (pair + 1) % 30 == 0:
                handle.flush()
                os.fsync(handle.fileno())
                atomic_json(
                    store / "checkpoints/STAGE2_SPARSE_CURSOR.json",
                    {
                        "schema": "ddm_qx3_sparse_cursor.v1",
                        "cursor": pair + 1,
                        "records": observed_mismatches,
                        "partial": {"path": str(sparse_partial), "bytes": sparse_partial.stat().st_size},
                    },
                )
    if observed_mismatches != mismatch_count:
        raise QX3Error("sparse correction mismatch denominator drifted")
    os.replace(sparse_partial, sparse_path)
    del derived, target
    stage = {
        "schema": "ddm_qx3_correction_forms.v1",
        "complete": True,
        "mismatches": mismatch_count,
        "forms": {"dense_delta": fact(dense_path), "sparse_u32_delta": fact(sparse_path)},
        "all_materialized_correction_payloads_retained": True,
    }
    atomic_json(stage_path, stage)
    return {"dense_delta": dense_path, "sparse_u32_delta": sparse_path}


def decode_correction(raw: bytes, source_path: Path, output_path: Path) -> dict[str, Any]:
    if raw[:4] == b"QXD1":
        if len(raw) < DENSE_HEADER.size:
            raise QX3Error("dense correction is truncated")
        magic, version, pairs, height, width, source_sha, target_sha = DENSE_HEADER.unpack_from(raw)
        if (magic, version, pairs, height, width) != (b"QXD1", 1, N_PAIRS, HEIGHT, WIDTH):
            raise QX3Error("dense correction identity drifted")
        if len(raw) != DENSE_HEADER.size + SITES:
            raise QX3Error("dense correction length drifted")
        codes = memoryview(raw)[DENSE_HEADER.size :]
        form = "dense_delta"
    elif raw[:4] == b"QXS1":
        if len(raw) < SPARSE_HEADER.size:
            raise QX3Error("sparse correction is truncated")
        magic, version, pairs, height, width, count, source_sha, target_sha = SPARSE_HEADER.unpack_from(raw)
        if (magic, version, pairs, height, width) != (b"QXS1", 1, N_PAIRS, HEIGHT, WIDTH):
            raise QX3Error("sparse correction identity drifted")
        if len(raw) != SPARSE_HEADER.size + count * 5:
            raise QX3Error("sparse correction length drifted")
        codes = memoryview(raw)[SPARSE_HEADER.size :]
        form = "sparse_u32_delta"
    else:
        raise QX3Error("unknown correction form")
    if sha256_file(source_path) != source_sha.hex():
        raise QX3Error("correction source baseline identity drifted")
    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    output[:] = source
    if form == "dense_delta":
        dense_codes = np.frombuffer(codes, dtype=np.uint8).reshape(N_PAIRS, HEIGHT, WIDTH)
        for pair in range(N_PAIRS):
            output[pair] = apply_delta_codes(np.asarray(source[pair]), dense_codes[pair])
    else:
        record_dtype = np.dtype([("index", ">u4"), ("code", "u1")])
        records = np.frombuffer(codes, dtype=record_dtype)
        indices = records["index"].astype(np.int64)
        if np.any(indices[1:] <= indices[:-1]) or np.any(indices >= SITES):
            raise QX3Error("sparse correction indices are noncanonical")
        flat = output.reshape(-1)
        source_flat = source.reshape(-1)
        record_codes = records["code"]
        if np.any((record_codes == 0) | (record_codes > 4)):
            raise QX3Error("sparse correction alphabet drifted")
        ranks = record_codes - 1
        flat[indices] = ranks + (ranks >= source_flat[indices]).astype(np.uint8)
    output.flush()
    del output, source
    os.replace(partial, output_path)
    if sha256_file(output_path) != target_sha.hex():
        raise QX3Error("correction receiver did not reconstruct its bound target baseline")
    return {"form": form, "output": fact(output_path), "target_sha256": target_sha.hex()}


def closure_raw(form: str, correction_raw: bytes, event_raw: bytes) -> bytes:
    return CLOSURE_HEADER.pack(
        b"QXR1",
        1,
        FORM_IDS[form],
        len(correction_raw),
        len(event_raw),
        bytes.fromhex(sha256_bytes(correction_raw)),
        bytes.fromhex(sha256_bytes(event_raw)),
    ) + correction_raw + event_raw


def parse_closure(raw: bytes) -> tuple[str, bytes, bytes]:
    if len(raw) < CLOSURE_HEADER.size:
        raise QX3Error("closure section is truncated")
    magic, version, form_id, correction_len, event_len, correction_sha, event_sha = CLOSURE_HEADER.unpack_from(raw)
    if magic != b"QXR1" or version != 1 or form_id not in FORM_NAMES:
        raise QX3Error("closure section identity drifted")
    offset = CLOSURE_HEADER.size
    if offset + correction_len + event_len != len(raw):
        raise QX3Error("closure section lengths do not close")
    correction = raw[offset : offset + correction_len]
    event = raw[offset + correction_len :]
    if sha256_bytes(correction) != correction_sha.hex() or sha256_bytes(event) != event_sha.hex():
        raise QX3Error("closure section inner payload integrity failed")
    return FORM_NAMES[form_id], correction, event


def retain_closure_race(
    store: Path, forms: Mapping[str, Path], event_raw: bytes
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage_path = store / "checkpoints/STAGE2_CLOSURE_RACE.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        return stage["best"], stage["candidates"]
    candidates: list[dict[str, Any]] = []
    for form, correction_path in forms.items():
        correction = correction_path.read_bytes()
        raw = closure_raw(form, correction, event_raw)
        root = store / "retained/closure_candidates" / form
        raw_path = root / "raw.bin"
        atomic_bytes(raw_path, raw)
        coder_rows: list[dict[str, Any]] = []
        for codec in CODECS:
            coded = compress(codec, raw)
            repeat = compress(codec, raw)
            if coded != repeat or decompress(codec, coded) != raw:
                raise QX3Error(f"{form}/{codec} failed deterministic real-coder parse-back")
            path = root / f"candidate.{codec}.bin"
            repeat_path = root / f"candidate.{codec}.repeat.bin"
            atomic_bytes(path, coded)
            atomic_bytes(repeat_path, repeat)
            coder_rows.append(
                {
                    "codec": codec,
                    "payload": fact(path),
                    "repeat": fact(repeat_path),
                    "deterministic_repeat": True,
                    "parseback_exact": True,
                }
            )
        winner = min(coder_rows, key=lambda row: (row["payload"]["bytes"], CODECS[row["codec"]]))
        candidates.append(
            {
                "form": form,
                "raw": fact(raw_path),
                "correction_raw": fact(correction_path),
                "coders": coder_rows,
                "winner_codec": winner["codec"],
                "winner_payload": winner["payload"],
                "section_cap_bytes": SECTION_CAP_BYTES,
                "delta_bytes_vs_section_cap": winner["payload"]["bytes"] - SECTION_CAP_BYTES,
                "incremental_bytes_vs_qx2_event_payload": winner["payload"]["bytes"] - QX2_EVENT_CODED_BYTES,
            }
        )
    best = min(
        candidates,
        key=lambda row: (row["winner_payload"]["bytes"], FORM_IDS[row["form"]]),
    )
    stage = {
        "schema": "ddm_qx3_closure_race.v1",
        "complete": True,
        "axis": AXIS,
        "best": best,
        "candidates": candidates,
        "all_raw_forms_coder_candidates_and_repeats_retained": True,
    }
    atomic_json(stage_path, stage)
    return best, candidates


def build_complete_packet(core: bytes, raw: bytes, coded: bytes, codec: str) -> bytes:
    records, _sections, _codecs = parse_qxe(core, 7)
    section = QXE_SECTION.pack(
        8,
        CODECS[codec],
        0,
        len(raw),
        len(coded),
        bytes.fromhex(sha256_bytes(raw)),
        zlib.crc32(coded) & 0xFFFFFFFF,
    ) + coded
    return QXE_HEADER.pack(b"QXE1", 1, 0, 8) + b"".join(records) + section


def deterministic_archive(packet: bytes) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        info = zipfile.ZipInfo("state/qx1.qxe", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet)
    return output.getvalue()


def reconstruct_target(
    baseline_path: Path, events: Sequence[PartitionEvent], output_path: Path
) -> dict[str, Any]:
    by_pair: list[list[PartitionEvent]] = [[] for _ in range(N_PAIRS)]
    for event in events:
        by_pair[event.pair].append(event)
    baseline = np.memmap(baseline_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    for pair in range(N_PAIRS):
        output[pair] = baseline[pair]
        for event in by_pair[pair]:
            if int(output[pair, event.row, event.col]) != event.baseline_class:
                raise QX3Error("decoded event baseline class differs at application time")
            output[pair, event.row, event.col] = event.target_class
        if (pair + 1) % 30 == 0:
            output.flush()
    output.flush()
    del output, baseline
    os.replace(partial, output_path)
    observed = fact(output_path)
    if observed["sha256"] != QX2_TARGET_SHA256:
        raise QX3Error("receiver output does not match the retained S2 target semantic SHA")
    return observed


def receiver_decode(
    archive_bytes: bytes,
    store: Path,
    qx2: Any,
    expected_events: Sequence[PartitionEvent],
    repeat: bool,
) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(archive_bytes), "r") as archive:
        if archive.namelist() != ["state/qx1.qxe"]:
            raise QX3Error("QX3 archive member roster drifted")
        packet = archive.read("state/qx1.qxe")
    records, sections, _codecs = parse_qxe(packet, 8)
    core = QXE_HEADER.pack(b"QXE1", 1, 0, 7) + b"".join(records[:7])
    if sha256_bytes(core) != PINS[QX1_CORE]:
        raise QX3Error("QX3 archive core differs from pinned QX1 core")
    form, correction_raw, event_raw = parse_closure(sections[8])
    derived_path, _stage = derive_decoder_baseline(store, core)
    suffix = ".repeat" if repeat else ""
    corrected_path = store / f"retained/receiver/corrected_baseline{suffix}.u8"
    correction = decode_correction(correction_raw, derived_path, corrected_path)
    events = tuple(qx2.decode_boundary_enumerative(event_raw, corrected_path))
    if len(events) != EVENTS or tuple(events) != tuple(expected_events):
        raise QX3Error("archive receiver did not reconstruct all 17,926 QX2 events exactly")
    target_path = store / f"retained/receiver/reconstructed_target{suffix}.u8"
    target = reconstruct_target(corrected_path, events, target_path)
    return {
        "form": form,
        "correction": correction,
        "events_decoded": len(events),
        "exact_event_identity": True,
        "target": target,
        "receiver_inputs": ["archive.zip", "generic QX1/QBT/QX2 receiver code"],
        "encoder_only_inputs_used_by_receiver": [],
    }


def preflight(store: Path) -> tuple[dict[str, Any], Any, tuple[PartitionEvent, ...]]:
    if store.resolve() != STORE.resolve():
        raise QX3Error(f"custody is pinned to {STORE}, not {store.resolve()}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise QX3Error(f"storage preflight failed: {free} < {MINIMUM_FREE_BYTES}")
    inputs = {str(path): require_fact(path, digest, path.name) for path, digest in PINS.items()}
    qx2_result = json.loads(QX2_RESULT.read_text(encoding="utf-8"))
    qx2_complete = Path(qx2_result["best"]["complete_packet"]["path"])
    qx2_event_raw = Path(qx2_result["best"]["raw"]["path"])
    inputs[str(qx2_complete)] = require_fact(
        qx2_complete, qx2_result["best"]["complete_packet"]["sha256"], "QX2 complete packet"
    )
    inputs[str(qx2_event_raw)] = require_fact(
        qx2_event_raw, qx2_result["best"]["raw"]["sha256"], "QX2 event raw"
    )
    qx2 = importlib.import_module("experiments.ddm_qx2_events_section_redesign")
    seed = decode_partition_seed(EVENT_SOURCE.read_bytes())
    expected_events = tuple(sorted(seed.events))
    if (seed.n_pairs, seed.height, seed.width, len(expected_events)) != (
        N_PAIRS,
        HEIGHT,
        WIDTH,
        EVENTS,
    ):
        raise QX3Error("retained S2 event geometry drifted")
    stage = {
        "schema": "ddm_qx3_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "selection_mode": "full n600",
        "storage": {"path": str(store), "observed_free_bytes": free},
        "inputs": inputs,
        "denominators": {"pairs": N_PAIRS, "sites": SITES, "events": EVENTS},
        "source_boundary": {
            "encoder_only": [
                "QX2 retained C1 conditioning baseline (correction target)",
                "retained S2 event object and GT cache (verification only)",
            ],
            "receiver_available": [
                "QX1 core QXE sections",
                "counted closure section",
                "generic deterministic decoder code",
            ],
        },
        "scorers_loaded": 0,
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
        "metal_invocations": 0,
    }
    atomic_json(store / "checkpoints/STAGE0_INPUT_TRACE.json", stage)
    return stage, qx2, expected_events


def run(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage0, qx2, expected_events = preflight(store)
    core = QX1_CORE.read_bytes()
    derived_path, derive_stage = derive_decoder_baseline(store, core)
    comparison = characterize_mismatch(store, derived_path, QX2_BASELINE, expected_events)
    mismatch_count = int(comparison["full_field"]["mismatches"])
    forms = build_correction_forms(store, derived_path, QX2_BASELINE, mismatch_count)
    qx2_result = json.loads(QX2_RESULT.read_text(encoding="utf-8"))
    event_raw = Path(qx2_result["best"]["raw"]["path"]).read_bytes()
    best, candidates = retain_closure_race(store, forms, event_raw)
    raw = Path(best["raw"]["path"]).read_bytes()
    coded = Path(best["winner_payload"]["path"]).read_bytes()
    packet = build_complete_packet(core, raw, coded, best["winner_codec"])
    packet_repeat = build_complete_packet(core, raw, coded, best["winner_codec"])
    archive = deterministic_archive(packet)
    archive_repeat = deterministic_archive(packet_repeat)
    if packet != packet_repeat or archive != archive_repeat:
        raise QX3Error("complete packet/archive repeat drifted")
    packet_path = store / "retained/complete/complete.qxe"
    packet_repeat_path = store / "retained/complete/complete.repeat.qxe"
    archive_path = store / "retained/complete/archive.zip"
    archive_repeat_path = store / "retained/complete/archive.repeat.zip"
    atomic_bytes(packet_path, packet)
    atomic_bytes(packet_repeat_path, packet_repeat)
    atomic_bytes(archive_path, archive)
    atomic_bytes(archive_repeat_path, archive_repeat)
    primary_receiver = receiver_decode(archive, store, qx2, expected_events, repeat=False)
    repeat_receiver = receiver_decode(archive_repeat, store, qx2, expected_events, repeat=True)
    if primary_receiver["target"]["sha256"] != repeat_receiver["target"]["sha256"]:
        raise QX3Error("receiver output repeat drifted")
    archive_fact = fact(archive_path)
    gate_cleared = archive_fact["bytes"] < ARCHIVE_GATE_EXCLUSIVE
    pure_derivation = mismatch_count == 0
    verdict = "RECEIVER-CLOSED" if gate_cleared else "BLOCKED"
    blocker = None
    if not gate_cleared:
        blocker = {
            "name": "QX1_QBT_BASELINE_DIFF_REQUIRES_OVER_CAP_COUNTED_CORRECTION",
            "missing_input": (
                "a decoder-available statistic that reproduces QX2's GT-derived C1 conditioning baseline; "
                "QX1 carries only the approximate QBT field"
            ),
            "cheapest_measured_route": {
                "form": best["form"],
                "codec": best["winner_codec"],
                "closure_section_bytes": best["winner_payload"]["bytes"],
                "section_cap_bytes": SECTION_CAP_BYTES,
                "excess_bytes": best["winner_payload"]["bytes"] - SECTION_CAP_BYTES,
            },
        }
    result = {
        "schema": "ddm_qx3_receiver_closure.v1",
        "complete": True,
        "verdict": verdict,
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "selection_mode": "full n600",
        "denominators": {"pairs": N_PAIRS, "sites": SITES, "events": EVENTS},
        "stage0": stage0,
        "derived_baseline": derive_stage,
        "bit_comparison": comparison,
        "pure_decoder_derivation_exact": pure_derivation,
        "closure_candidates": candidates,
        "best": best,
        "complete_packet": fact(packet_path),
        "complete_packet_repeat": fact(packet_repeat_path),
        "archive": archive_fact,
        "archive_repeat": fact(archive_repeat_path),
        "archive_gate": {
            "strict_archive_bytes_lt": ARCHIVE_GATE_EXCLUSIVE,
            "observed_archive_bytes": archive_fact["bytes"],
            "cleared": gate_cleared,
            "delta_bytes_vs_gate": archive_fact["bytes"] - (ARCHIVE_GATE_EXCLUSIVE - 1),
            "qx1_core_archive_bytes": QX1_CORE_ARCHIVE_BYTES,
            "section_cap_bytes": SECTION_CAP_BYTES,
        },
        "receiver_primary": primary_receiver,
        "receiver_repeat": repeat_receiver,
        "exact_receiver_decode": True,
        "deterministic_repeat": True,
        "blocker": blocker,
        "authority_boundaries": {
            "scorers_loaded": 0,
            "contest_eval_invocations": 0,
            "modal_invocations": 0,
            "metal_invocations": 0,
            "distortion_measured": False,
            "contest_score_measured": False,
            "rate_and_receiver_exactness_measured": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "RESULT.json", result)
    manifest = {
        "schema": "ddm_qx3_run_manifest.v1",
        "complete": True,
        "result": fact(store / "RESULT.json"),
        "command": f"{sys.executable} {Path(__file__).resolve()} --resume-from {store}",
        "source": fact(Path(__file__).resolve()),
        "retention": "all derived fields, correction forms, closure raws, real-coder candidates, repeats, packets, archives, and receiver outputs retained",
        "cleanup": "none fired; all experiment payloads are durable under AP custody",
    }
    atomic_json(store / "RUN_MANIFEST.json", manifest)
    atomic_json(
        store / "checkpoints/STAGE3_COMPLETE.json",
        {
            "schema": "ddm_qx3_complete.v1",
            "complete": True,
            "verdict": verdict,
            "result": fact(store / "RESULT.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, default=STORE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.resume_from)
    print(json.dumps({"verdict": result["verdict"], "archive": result["archive"], "blocker": result["blocker"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
