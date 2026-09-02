#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bind QXO1's exact archive to retained BR2-format scorer inputs.

The default ``prepare`` and ``fire-order`` actions are scorer-free.  They
decode the exact QXO1 archive, reproduce its overwrite field, prove which
counted QBT state determines the renderer, and retain the camera-grid payloads
that a later MAIN-owned scorer run will consume.  ``score`` is deliberately a
separate, launch-authorized action: it imports the frozen scorers only after an
active MAIN lane claim has been verified.

This runner never treats a semantic field as RGB.  The current QXO1 contract
does not feed section 8 into the inherited QBT renderer; that causal boundary
is recorded beside the payload rather than hidden by an invented palette.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import platform
import shutil
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qx2_events_section_redesign as qx2
from experiments import ddm_qxo1_target_overwrite_grammar as qxo1

SCHEMA: Final = "ddm_qxr1_qxo1_realization_binding.v1"
FIRE_SCHEMA: Final = "ddm_qxr1_qxo1_scorer_fire_order.v1"
SCORE_SCHEMA: Final = "ddm_qxr1_qxo1_scorer_result.v1"
AXIS: Final = "[scorer-free exact receiver/render-input binding]"
SCORE_AXIS: Final = "[macOS-CPU advisory]"
N: Final = 600
H: Final = 384
W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1_164
SITES: Final = N * H * W
CHUNK_PAIRS: Final = 30
RATE_DENOMINATOR: Final = 37_545_489
ARCHIVE_BYTES: Final = 129_309
ARCHIVE_SHA256: Final = "2487f5150fd3c38087fb5ada48d00e953c7d88a8a7219e29fbf53420657bb07f"
COMPLETE_QXE_SHA256: Final = "2308820b56b29abef69556bbd98e12758cdf7e3adc6f214fe38b83ab0066a6d6"
CORE_SHA256: Final = "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95"
BASELINE_SHA256: Final = "afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd"
FIELD_SHA256: Final = "9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7"
GRAMMAR_SHA256: Final = "bb6c1b8626f06632ee1b3f2d6088a25d85e6d7db3c4d00b258686418b67c85ea"
GRAMMAR_CODED_SHA256: Final = "b0c68d2226febf336521d454fa13a9c0fa324a14d2b1cb14ab54038b89de34f2"
OVERWRITE_RECORDS_SHA256: Final = "13e5b7419a1873c6543075d1fde4347644247fae25fc46d281450ad244cd2ee9"
MODEL_SHA256: Final = "2280c2d3c54d1781559ec130123a05ec664dbdf347b04f379805bfbe67f59085"
LATENT_META_SHA256: Final = "79128a18dec7177dcd9b6922f261f1f6dc3b637b27d04c39baae8c4fed0af2b2"
LATENTS_SHA256: Final = "ff7db019f3d774da8abf20a79c9bba4df7b2b73d277ac09ba4feedd0505df9d2"
POSE_STREAM_SHA256: Final = "9142ab46a65d7ef9b62bcf98d789ea9741212f163d16940fe284a3786e16bf4b"
QXO_RESULT_SHA256: Final = "b7b9dd4fb1dbb70aa6dd41a32a6b998c30588103c0d2a8184d71c6ff9147a80a"
BR2_RESULT_SHA256: Final = "a7ae997a75cd86fa1e36552cd83c5b7b208874438832ebc1555e24666e9a4c8e"
BR2_PACKET_SHA256: Final = "8c26684d33313ca44f3d4f02cf3c369f0f33d6de37eeba42ae4220faed3e6d38"
EXPECTED_EVENTS: Final = 17_926
EXPECTED_NOOPS: Final = 9_177
EXPECTED_WRITES: Final = 8_749
EXPECTED_GRAMMAR_CODED_BYTES: Final = 15_417
EXPECTED_CORE_ARCHIVE_BYTES: Final = 113_844
EXPECTED_CORE_QXE_BYTES: Final = 113_720
AFR1_SCORE: Final = 0.14797617125559104
AFR1_BYTES: Final = 180_002
AFR1_ARCHIVE_SHA256: Final = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
MIN_FREE_PREPARE: Final = 2_500_000_000
MIN_FREE_SCORE: Final = 1_500_000_000

SOURCE_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar")
OUTPUT_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding")
QXO_RESULT: Final = SOURCE_ROOT / "RESULT.json"
QXO_ARCHIVE: Final = SOURCE_ROOT / "retained/grammar_v1/archive.zip"
QXO_BASELINE: Final = SOURCE_ROOT / "retained/derived/qx1_decoder_baseline.u8"
QXO_REFERENCE_FIELD: Final = SOURCE_ROOT / "retained/grammar_v1/decoded_field.primary.u8"
QXO_CODED_GRAMMAR: Final = SOURCE_ROOT / "retained/grammar_v1/candidate.brotli_q11.bin"
BR2_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_br2")
BR2_RESULT: Final = BR2_ROOT / "REALIZED_RESULT.json"
BR2_PACKET: Final = BR2_ROOT / "inputs/packet.qbf"
ACTIVE_CLAIMS: Final = REPO / ".omx/state/active_lane_dispatch_claims.md"
CANONICAL_POINTER: Final = REPO / ".omx/state/canonical_frontier_pointer.json"
CLAIM_ID: Final = "ddm_qxr1_qxo1_scorer_20260902"
QBF_SOURCE_SHA256: Final = "cdf90d1a4d7d13001118f50a76692c04605f8e5ae9a7816c80f6e346160c7b9c"
QX2_SOURCE_SHA256: Final = "88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c"
QXO_SOURCE_SHA256: Final = "d71be64b20083593ff9615b55d232ec9ac753b03d24941921a5b69b613cf08c0"

QXE_HEADER: Final = struct.Struct(">4sBBH")
QXE_SECTION: Final = struct.Struct(">BBHII32sI")
QXT_HEADER: Final = struct.Struct(">4sBBB")
QBT_TENSOR_HEADER: Final = struct.Struct(">BBBBfII")
CODEC_NAMES: Final = {1: "brotli_q11", 2: "lzma9e", 3: "zlib9"}
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


class QXR1Error(RuntimeError):
    """A QXR1 custody, receiver, causal-binding, or lane gate failed."""


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QXR1Error(f"required file is absent: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def require_fact(path: Path, digest: str, *, size: int | None = None) -> dict[str, Any]:
    observed = file_fact(path)
    if observed["sha256"] != digest or (size is not None and observed["bytes"] != size):
        raise QXR1Error(f"frozen input drifted: {path}: {observed}")
    return observed


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return file_fact(path)


def atomic_copy(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with source.open("rb") as reader, partial.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(partial, target)
    return file_fact(target)


def atomic_json(path: Path, payload: Any) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return atomic_bytes(path, encoded)


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return file_fact(path)


def storage_preflight(output: Path, *, required: int) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise QXR1Error(f"output must be the chartered AP store: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < required:
        raise QXR1Error(f"AP storage preflight refused: free={usage.free}, required={required}")
    return {
        "root": str(output.resolve()),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; this runner never deletes retained payloads",
    }


def decompress(codec: str, payload: bytes) -> bytes:
    if codec == "brotli_q11":
        return brotli.decompress(payload)
    if codec == "lzma9e":
        return lzma.decompress(payload)
    if codec == "zlib9":
        return zlib.decompress(payload)
    raise QXR1Error(f"unsupported QXE codec: {codec}")


def parse_qxe(packet: bytes, count: int) -> tuple[list[bytes], dict[int, bytes], dict[int, str]]:
    if len(packet) < QXE_HEADER.size or QXE_HEADER.unpack_from(packet) != (b"QXE1", 1, 0, count):
        raise QXR1Error("QXE identity or section count drifted")
    records: list[bytes] = []
    sections: dict[int, bytes] = {}
    codecs: dict[int, str] = {}
    offset = QXE_HEADER.size
    for expected_id in range(1, count + 1):
        start = offset
        if offset + QXE_SECTION.size > len(packet):
            raise QXR1Error("QXE section header is truncated")
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, crc = QXE_SECTION.unpack_from(packet, offset)
        offset += QXE_SECTION.size
        end = offset + coded_len
        if section_id != expected_id or codec_id not in CODEC_NAMES or reserved or end > len(packet):
            raise QXR1Error("QXE section envelope drifted")
        coded = packet[offset:end]
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_len or sha256_bytes(raw) != raw_sha.hex() or zlib.crc32(coded) & 0xFFFFFFFF != crc:
            raise QXR1Error("QXE section integrity failed")
        records.append(packet[start:end])
        sections[section_id] = raw
        codecs[section_id] = CODEC_NAMES[codec_id]
        offset = end
    if offset != len(packet):
        raise QXR1Error("QXE packet has trailing bytes")
    return records, sections, codecs


def reassemble_model(groups: Sequence[bytes]) -> bytes:
    expected = ((1, 28), (2, 10), (3, 4))
    tensors: dict[str, bytes] = {}
    for raw, (group_id, tensor_count) in zip(groups, expected, strict=True):
        if len(raw) < QXT_HEADER.size or QXT_HEADER.unpack_from(raw) != (b"QXT1", group_id, 0, tensor_count):
            raise QXR1Error("QXT model-group identity drifted")
        offset = QXT_HEADER.size
        for _ in range(tensor_count):
            start = offset
            if offset + QBT_TENSOR_HEADER.size > len(raw):
                raise QXR1Error("QXT tensor header is truncated")
            name_len, _bits, ndim, reserved, _scale, count, packed_len = QBT_TENSOR_HEADER.unpack_from(raw, offset)
            offset += QBT_TENSOR_HEADER.size
            end_name = offset + name_len
            end_shape = end_name + 2 * ndim
            end_record = end_shape + packed_len
            shape = tuple(struct.unpack_from(">H", raw, end_name + 2 * index)[0] for index in range(ndim))
            name = raw[offset:end_name].decode("ascii")
            if reserved or not name or math.prod(shape) != count or end_record > len(raw) or name in tensors:
                raise QXR1Error("QXT tensor record drifted")
            tensors[name] = raw[start:end_record]
            offset = end_record
        if offset != len(raw):
            raise QXR1Error("QXT model group has trailing bytes")
    model = b"QBT1" + struct.pack(">H", len(tensors)) + b"".join(tensors[name] for name in sorted(tensors))
    if sha256_bytes(model) != MODEL_SHA256:
        raise QXR1Error("reassembled QBT model differs from the pinned ancestor")
    qbf1.decode_model(model)
    return model


def archive_sections(archive_payload: bytes) -> tuple[bytes, dict[int, bytes], dict[str, Any]]:
    with zipfile.ZipFile(BytesIO(archive_payload), "r") as archive:
        if archive.namelist() != ["state/qx1.qxe"]:
            raise QXR1Error("QXO1 archive member roster drifted")
        packet = archive.read("state/qx1.qxe")
    if sha256_bytes(packet) != COMPLETE_QXE_SHA256:
        raise QXR1Error("QXO1 complete QXE SHA-256 drifted")
    records, sections, codecs = parse_qxe(packet, 8)
    core = QXE_HEADER.pack(b"QXE1", 1, 0, 7) + b"".join(records[:7])
    if len(core) != EXPECTED_CORE_QXE_BYTES or sha256_bytes(core) != CORE_SHA256:
        raise QXR1Error("QXO1 seven-section core identity drifted")
    model = reassemble_model((sections[2], sections[3], sections[4]))
    trace = {
        "complete_qxe_sha256": sha256_bytes(packet),
        "core_sha256": sha256_bytes(core),
        "core_qxe_bytes": len(core),
        "core_archive_bytes": EXPECTED_CORE_ARCHIVE_BYTES,
        "grammar_raw_sha256": sha256_bytes(sections[8]),
        "grammar_raw_bytes": len(sections[8]),
        "model_sha256": sha256_bytes(model),
        "latent_meta_sha256": sha256_bytes(sections[5]),
        "latents_sha256": sha256_bytes(sections[6]),
        "pose_stream_sha256": sha256_bytes(sections[7]),
        "section_codecs": {str(key): value for key, value in sorted(codecs.items())},
    }
    expected = {
        "grammar_raw_sha256": GRAMMAR_SHA256,
        "latent_meta_sha256": LATENT_META_SHA256,
        "latents_sha256": LATENTS_SHA256,
        "pose_stream_sha256": POSE_STREAM_SHA256,
    }
    if any(trace[key] != value for key, value in expected.items()):
        raise QXR1Error(f"QXO1 section trace drifted: {trace}")
    return core, sections, trace


def qbt_render_binding(sections: Mapping[int, bytes]) -> dict[str, Any]:
    packet_fact = require_fact(BR2_PACKET, BR2_PACKET_SHA256)
    decoded = qbf1.decode_packet(BR2_PACKET.read_bytes())
    qxo_model = reassemble_model((sections[2], sections[3], sections[4]))
    comparisons = {
        "model": sha256_bytes(qxo_model) == sha256_bytes(decoded.sections[qbf1.SECTION_MODEL]),
        "latent_meta": sections[5] == decoded.sections[qbf1.SECTION_LATENT_META],
        "latents": sections[6] == decoded.sections[qbf1.SECTION_LATENTS],
    }
    if not all(comparisons.values()):
        raise QXR1Error(f"QXO1 renderer-determining state differs from BR2 QBT: {comparisons}")
    return {
        "br2_packet": packet_fact,
        "renderer_state_byte_identity": comparisons,
        "renderer_inputs_equal": True,
        "qxo_section_8_consumed_by_qbt_renderer": False,
        "qxo_section_7_pose_stream_consumed_by_qbt_renderer": False,
        "causal_conclusion": (
            "QXO1 section 8 changes the decoded semantic field and section 7 carries a pose stream, but the "
            "current inherited QBT RGB renderer reads only model, latent-meta, and latent sections. The retained "
            "BR2 camera payload is therefore an exact scorer-input materialization of the current QXO1 renderer "
            "state, not a distortion transfer."
        ),
    }


def field_diff(baseline_path: Path, field_path: Path) -> dict[str, Any]:
    baseline = np.memmap(baseline_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    transitions: Counter[str] = Counter()
    source_classes: Counter[str] = Counter()
    target_classes: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    per_pair = np.zeros(N, dtype=np.int64)
    first_rows: list[dict[str, int | str]] = []
    for pair in range(N):
        before = np.asarray(baseline[pair])
        after = np.asarray(field[pair])
        coords = np.argwhere(before != after)
        per_pair[pair] = len(coords)
        for row, col in coords:
            source = int(before[row, col])
            target = int(after[row, col])
            transition = f"{source}:{CLASS_NAMES[source]}->{target}:{CLASS_NAMES[target]}"
            vertical = min(2, int(row) * 3 // H)
            horizontal = min(2, int(col) * 3 // W)
            region = f"v{vertical}_h{horizontal}"
            transitions[transition] += 1
            source_classes[f"{source}:{CLASS_NAMES[source]}"] += 1
            target_classes[f"{target}:{CLASS_NAMES[target]}"] += 1
            regions[region] += 1
            if len(first_rows) < 64:
                first_rows.append(
                    {"pair": pair, "row": int(row), "col": int(col), "source": source, "target": target, "region": region}
                )
    mismatches = int(per_pair.sum())
    del baseline, field
    if mismatches != EXPECTED_WRITES or sum(transitions.values()) != EXPECTED_WRITES:
        raise QXR1Error(f"field mutation denominator drifted: {mismatches}")
    return {
        "denominator_sites": SITES,
        "mutations": mismatches,
        "mutation_fraction": mismatches / SITES,
        "source_class_counts": dict(sorted(source_classes.items())),
        "target_class_counts": dict(sorted(target_classes.items())),
        "transition_counts": dict(sorted(transitions.items())),
        "region_definition": "3x3 equal normalized image bins: v0 top/v2 bottom; h0 left/h2 right",
        "region_counts": dict(sorted(regions.items())),
        "per_pair": {
            "min": int(per_pair.min()),
            "max": int(per_pair.max()),
            "mean": float(per_pair.mean()),
            "zero_mutation_pairs": int(np.count_nonzero(per_pair == 0)),
            "counts": per_pair.tolist(),
        },
        "first_64": first_rows,
    }


def validate_render_chunk(
    path: Path,
    *,
    expected_ids: Sequence[int],
    expected_camera: np.ndarray,
    baseline: np.memmap,
    field: np.memmap,
) -> dict[str, Any]:
    required = {
        "pair_ids_i64",
        "camera_pair_u8",
        "qbt_baseline_class_u8",
        "qxo1_overwrite_class_u8",
        "qxo1_mutation_mask_u8",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise QXR1Error(f"prepared render-input payload roster drifted: {path}")
        ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
        camera = np.asarray(payload["camera_pair_u8"], dtype=np.uint8)
        before = np.asarray(payload["qbt_baseline_class_u8"], dtype=np.uint8)
        after = np.asarray(payload["qxo1_overwrite_class_u8"], dtype=np.uint8)
        mask = np.asarray(payload["qxo1_mutation_mask_u8"], dtype=np.uint8)
    start, stop = expected_ids[0], expected_ids[-1] + 1
    if ids.tolist() != list(expected_ids) or camera.shape != (
        len(expected_ids),
        2,
        3,
        CAMERA_H,
        CAMERA_W,
    ):
        raise QXR1Error(f"prepared render-input pair/camera geometry drifted: {path}")
    if not np.array_equal(camera, expected_camera):
        raise QXR1Error(f"prepared camera payload differs from the pinned BR2 source: {path}")
    if not np.array_equal(before, baseline[start:stop]) or not np.array_equal(after, field[start:stop]):
        raise QXR1Error(f"prepared render-input field payload drifted: {path}")
    if not np.array_equal(mask, (before != after).astype(np.uint8)):
        raise QXR1Error(f"prepared render-input mutation mask drifted: {path}")
    return file_fact(path)


def retain_render_chunks(output: Path, baseline_path: Path, field_path: Path) -> list[dict[str, Any]]:
    require_fact(BR2_RESULT, BR2_RESULT_SHA256)
    br2_result = json.loads(BR2_RESULT.read_text())
    sources = br2_result["retained_chunks"]
    if len(sources) != math.ceil(N / CHUNK_PAIRS):
        raise QXR1Error("BR2 retained chunk count drifted")
    baseline = np.memmap(baseline_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    retained: list[dict[str, Any]] = []
    for index, row in enumerate(sources):
        source = Path(row["path"])
        if file_fact(source) != row:
            raise QXR1Error(f"BR2 retained scorer-input chunk drifted: {source}")
        start = index * CHUNK_PAIRS
        stop = min(N, start + CHUNK_PAIRS)
        ids_expected = list(range(start, stop))
        with np.load(source, allow_pickle=False) as payload:
            ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
            camera = np.asarray(payload["camera_pair_u8"], dtype=np.uint8)
        if ids.tolist() != ids_expected:
            raise QXR1Error("BR2 chunk pair order drifted")
        target = output / "retained/render_inputs" / f"qxo1_pairs_{start:04d}_{stop - 1:04d}.npz"
        if target.is_file():
            retained.append(
                validate_render_chunk(
                    target,
                    expected_ids=ids_expected,
                    expected_camera=camera,
                    baseline=baseline,
                    field=field,
                )
            )
            continue
        fact = atomic_npz(
            target,
            pair_ids_i64=ids,
            camera_pair_u8=camera,
            qbt_baseline_class_u8=np.asarray(baseline[start:stop]),
            qxo1_overwrite_class_u8=np.asarray(field[start:stop]),
            qxo1_mutation_mask_u8=np.asarray(baseline[start:stop] != field[start:stop], dtype=np.uint8),
        )
        retained.append(
            validate_render_chunk(
                target,
                expected_ids=ids_expected,
                expected_camera=camera,
                baseline=baseline,
                field=field,
            )
        )
        atomic_json(
            output / "checkpoints" / f"stage_02_render_pairs_{start:04d}_{stop - 1:04d}.json",
            {"schema": "ddm_qxr1_render_chunk.v1", "complete": True, "source_br2_chunk": row, "payload": fact},
        )
        print(json.dumps({"prepared_pairs": stop, "n": N, "payload": fact}), flush=True)
    del baseline, field
    return retained


def prepare(output: Path) -> dict[str, Any]:
    started = time.time()
    storage = storage_preflight(output, required=MIN_FREE_PREPARE)
    source_facts = {
        "qxo_result": require_fact(QXO_RESULT, QXO_RESULT_SHA256),
        "qxo_archive": require_fact(QXO_ARCHIVE, ARCHIVE_SHA256, size=ARCHIVE_BYTES),
        "qxo_baseline": require_fact(QXO_BASELINE, BASELINE_SHA256, size=SITES),
        "qxo_reference_field": require_fact(QXO_REFERENCE_FIELD, FIELD_SHA256, size=SITES),
        "qxo_coded_grammar": require_fact(QXO_CODED_GRAMMAR, GRAMMAR_CODED_SHA256, size=EXPECTED_GRAMMAR_CODED_BYTES),
        "br2_result": require_fact(BR2_RESULT, BR2_RESULT_SHA256),
        "runner": file_fact(Path(__file__).resolve()),
        "qbf_receiver": require_fact(Path(qbf1.__file__).resolve(), QBF_SOURCE_SHA256),
        "qxo_receiver": require_fact(Path(qxo1.__file__).resolve(), QXO_SOURCE_SHA256),
        "qx2_conditioner": require_fact(Path(qx2.__file__).resolve(), QX2_SOURCE_SHA256),
    }
    archive_payload = QXO_ARCHIVE.read_bytes()
    core, sections, section_trace = archive_sections(archive_payload)
    inputs = output / "retained/inputs"
    retained_archive = atomic_bytes(inputs / "archive.zip", archive_payload)
    retained_core = atomic_bytes(inputs / "core.qxe", core)
    retained_grammar = atomic_bytes(inputs / "grammar.raw.qxo", sections[8])
    retained_model = atomic_bytes(inputs / "model.raw.qbt", reassemble_model((sections[2], sections[3], sections[4])))
    retained_latent_meta = atomic_bytes(inputs / "latent_meta.raw", sections[5])
    retained_latents = atomic_bytes(inputs / "latents.raw", sections[6])
    retained_pose = atomic_bytes(inputs / "pose_stream.raw", sections[7])
    retained_baseline = atomic_copy(QXO_BASELINE, output / "retained/receiver/qbt_baseline.u8")
    events = qxo1.decode_grammar(qx2, sections[8], Path(retained_baseline["path"]))
    if len(events) != EXPECTED_WRITES:
        raise QXR1Error("decoded overwrite count drifted")
    records = qxo1.serialize_overwrites(events)
    if sha256_bytes(records) != OVERWRITE_RECORDS_SHA256:
        raise QXR1Error("decoded overwrite-record payload drifted")
    retained_records = atomic_bytes(output / "retained/receiver/decoded_overwrites.u32be_target_u8", records)
    decoded_path = output / "retained/receiver/qxo1_decoded_field.u8"
    decoded_fact = qxo1.write_overwrite_output(Path(retained_baseline["path"]), events, decoded_path)
    if decoded_fact["sha256"] != FIELD_SHA256 or decoded_fact["bytes"] != SITES:
        raise QXR1Error("fresh QXO1 receiver output differs from the pinned decoded field")
    diff = field_diff(Path(retained_baseline["path"]), decoded_path)
    binding = qbt_render_binding(sections)
    render_chunks = retain_render_chunks(output, Path(retained_baseline["path"]), decoded_path)
    pointer = json.loads(CANONICAL_POINTER.read_text())
    result = {
        "schema": SCHEMA,
        "complete": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "score_claim": False,
        "scorers_loaded": 0,
        "selection_mode": "full n600",
        "n": N,
        "site_denominator": SITES,
        "archive": retained_archive,
        "resolved_counts": {
            "source_events": EXPECTED_EVENTS,
            "target_noops": EXPECTED_NOOPS,
            "actual_writes": len(events),
            "grammar_coded_bytes": EXPECTED_GRAMMAR_CODED_BYTES,
            "core_archive_bytes": EXPECTED_CORE_ARCHIVE_BYTES,
            "archive_bytes": ARCHIVE_BYTES,
            "largest_legal_archive_bytes": 137_985,
            "bytes_under_gate": 8_676,
        },
        "section_trace": section_trace,
        "retained_receiver_payloads": {
            "core": retained_core,
            "grammar": retained_grammar,
            "model": retained_model,
            "latent_meta": retained_latent_meta,
            "latents": retained_latents,
            "pose_stream": retained_pose,
            "baseline": retained_baseline,
            "overwrite_records": retained_records,
            "decoded_field": file_fact(decoded_path),
        },
        "field_diff": diff,
        "render_binding": binding,
        "retained_render_chunks": render_chunks,
        "render_chunk_pairs": CHUNK_PAIRS,
        "source_facts": source_facts,
        "storage_preflight": storage,
        "closed_form_first": (
            "Exact QXE parsing, tensor/latent byte identity, overwrite application, and deterministic renderer-state "
            "causality decide the binding; no fit, surrogate, sampling, or scorer estimate is used."
        ),
        "distortion_transfer": "REFUSED_IN_BOTH_DIRECTIONS",
        "prior_law_prediction": {
            "prediction": "realized d_seg remains order 0.17",
            "falsifier": "d_seg <= 0.01 and d_pose <= 1.25e-4 on this exact archive/input binding",
            "status": "UNTESTED_UNTIL_MAIN_SCORER_FIRE",
        },
        "current_frontier": {
            "expected_score": AFR1_SCORE,
            "expected_bytes": AFR1_BYTES,
            "expected_archive_sha256": AFR1_ARCHIVE_SHA256,
            "observed_effective_frontier": pointer.get("effective_frontier"),
        },
        "elapsed_seconds": time.time() - started,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "pointer_moved": False,
        "boundaries": [
            "No SegNet or PoseNet was imported by prepare.",
            "Sections 7 and 8 carry pose/semantic state but are not consumed by the current QBT RGB renderer.",
            "The queued scorer measurement is advisory and cannot promote the contest pointer.",
            "No BR2 distortion component is attached to QXO1 bytes by this receipt.",
        ],
    }
    atomic_json(output / "checkpoints/stage_03_prepare_complete.json", result)
    atomic_json(output / "BINDING_RESULT.json", result)
    return result


def emit_fire_order(output: Path) -> dict[str, Any]:
    result_path = output / "BINDING_RESULT.json"
    if not result_path.is_file():
        prepare(output)
    binding = json.loads(result_path.read_text())
    if not binding.get("complete") or binding["archive"]["sha256"] != ARCHIVE_SHA256:
        raise QXR1Error("a complete pinned binding receipt is required before fire-order emission")
    command = (
        "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python "
        "experiments/ddm_qxr1_qxo1_realization_binding.py score "
        f"--output {OUTPUT_ROOT} --resume-from {OUTPUT_ROOT} "
        f"--scorer-claim-id {CLAIM_ID} --launch-authorized"
    )
    order = {
        "schema": FIRE_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN n600 local scorer-realization scheduler",
        "consumer_store": str((output / "SCORER_RESULT.json").resolve()),
        "retention_root": str((output / "retained/scorer_outputs").resolve()),
        "axis": SCORE_AXIS,
        "promotable": False,
        "score_claim": False,
        "claim_id_to_append_as_a_fresh_active_local_scorer_row": CLAIM_ID,
        "fire_trigger": (
            "MAIN verifies the newest relevant scorer row is terminal, appends a fresh unique active local_macos_cpu "
            f"claim for {CLAIM_ID}, confirms no newer active scorer claim within 24 h, AP free bytes >= "
            f"{MIN_FREE_SCORE}, and re-matches archive {ARCHIVE_SHA256}, decoded field {FIELD_SHA256}, core "
            f"{CORE_SHA256}, grammar {GRAMMAR_SHA256}, plus every prepared render-input fact."
        ),
        "command": command,
        "expected_wall_seconds": 485,
        "chunk_pairs": CHUNK_PAIRS,
        "maximum_chunk_pairs": 120,
        "components": ["d_seg", "d_pose", "100*d_seg", "sqrt(10*d_pose)", "rate", "S"],
        "prediction": binding["prior_law_prediction"],
        "causal_warning": binding["render_binding"]["causal_conclusion"],
        "no_distortion_transfer": True,
        "all_payloads_retained": True,
    }
    atomic_json(output / "FIRE_ORDER.json", order)
    return order


def active_claim(claim_id: str) -> dict[str, Any]:
    if claim_id != CLAIM_ID:
        raise QXR1Error(f"claim id must be the sealed QXR1 id: {CLAIM_ID}")
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) == 8 and fields[0].startswith("20"):
            rows.append({"timestamp": fields[0], "lane_id": fields[2], "platform": fields[3], "status": fields[6], "raw": line})
    newest: dict[str, dict[str, str]] = {}
    for row in rows:
        newest.setdefault(row["lane_id"], row)
    own = newest.get(claim_id)
    if own is None or own["platform"] != "local_macos_cpu" or not own["status"].startswith("active_"):
        raise QXR1Error("MAIN must append a fresh active local_macos_cpu QXR1 scorer claim")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        if datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise QXR1Error(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "row": own["raw"], "registry": file_fact(ACTIVE_CLAIMS)}


def aggregate_score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seg_errors = sum(int(row["seg_errors"]) for row in rows)
    seg_pixels = sum(int(row["seg_pixels"]) for row in rows)
    pose_sse = sum(float(row["pose_sse"]) for row in rows)
    pose_values = sum(int(row["pose_values"]) for row in rows)
    if (seg_pixels, pose_values) != (SITES, N * 6):
        raise QXR1Error("scorer denominators differ from full n600")
    d_seg = seg_errors / seg_pixels
    d_pose = pose_sse / pose_values
    rate = 25.0 * ARCHIVE_BYTES / RATE_DENOMINATOR
    score = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + rate
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "seg_term": 100.0 * d_seg,
        "pose_term": math.sqrt(10.0 * d_pose),
        "rate": rate,
        "S": score,
        "seg_errors": seg_errors,
        "seg_pixels": seg_pixels,
        "pose_squared_error_sum": pose_sse,
        "pose_values": pose_values,
    }


def score(output: Path, *, resume_from: Path, claim_id: str, launch_authorized: bool) -> dict[str, Any]:
    if not launch_authorized:
        raise QXR1Error("MAIN scorer realization requires --launch-authorized")
    if resume_from.resolve() != output.resolve() or output.resolve() != OUTPUT_ROOT.resolve():
        raise QXR1Error("--resume-from and --output must name the chartered QXR1 store")
    claim = active_claim(claim_id)
    storage = storage_preflight(output, required=MIN_FREE_SCORE)
    binding = json.loads((output / "BINDING_RESULT.json").read_text())
    order = json.loads((output / "FIRE_ORDER.json").read_text())
    if binding["archive"]["sha256"] != ARCHIVE_SHA256 or order["command"].find(claim_id) < 0:
        raise QXR1Error("binding or sealed fire order drifted")

    # Scorer-bearing imports are intentionally below every launch/custody gate.
    import torch

    from experiments import ddm_qbt1_qbflow_trainer as qbt1
    from experiments import ddm_qbz1_descent_rate_configuration as qbz1
    from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
    from tac.scorer import load_differentiable_scorers

    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="QXR1 DALI partition")
    assert_gt_lineage(qbz1.GT_POSE6, required=AUTHORITY_LINEAGE, instrument="QXR1 DALI pose")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    torch.manual_seed(qbz1.SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    started = time.time()
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for source_row in binding["retained_render_chunks"]:
        source = Path(source_row["path"])
        if file_fact(source) != source_row:
            raise QXR1Error(f"prepared render input drifted: {source}")
        with np.load(source, allow_pickle=False) as payload:
            ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
            camera_u8 = np.asarray(payload["camera_pair_u8"], dtype=np.uint8)
        target = output / "retained/scorer_outputs" / f"scorer_pairs_{ids[0]:04d}_{ids[-1]:04d}.npz"
        if target.is_file():
            with np.load(target, allow_pickle=False) as payload:
                chunk_rows = json.loads(bytes(np.asarray(payload["pair_rows_json_u8"], dtype=np.uint8)).decode("utf-8"))
            rows.extend(chunk_rows)
            outputs.append(file_fact(target))
            continue
        with torch.no_grad():
            camera = torch.from_numpy(camera_u8.astype(np.float32))
            pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
            argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            pose = pose6.cpu().numpy().astype("<f4")
            logits_f16 = logits.cpu().numpy().astype("<f2")
        chunk_rows = []
        for index, pair_id in enumerate(ids.tolist()):
            chunk_rows.append(
                {
                    "pair_id": pair_id,
                    "seg_errors": int(np.count_nonzero(argmax[index] != gt[pair_id])),
                    "seg_pixels": H * W,
                    "pose_sse": float(np.square(pose[index].astype(np.float64) - pose_target[pair_id].astype(np.float64)).sum()),
                    "pose_values": 6,
                }
            )
        fact = atomic_npz(
            target,
            pair_ids_i64=ids,
            segnet_logits_f16=logits_f16,
            segnet_argmax_u8=argmax,
            target_argmax_u8=np.asarray(gt[ids], dtype=np.uint8),
            posenet_pose6_f32=pose,
            target_pose6_f32=np.asarray(pose_target[ids], dtype="<f4"),
            pair_rows_json_u8=np.frombuffer(json.dumps(chunk_rows, sort_keys=True).encode("utf-8"), dtype=np.uint8),
        )
        rows.extend(chunk_rows)
        outputs.append(fact)
        atomic_json(
            output / "checkpoints" / f"stage_04_scorer_pairs_{ids[0]:04d}_{ids[-1]:04d}.json",
            {"schema": "ddm_qxr1_scorer_chunk.v1", "complete": True, "render_input": source_row, "payload": fact},
        )
        print(json.dumps({"scored_pairs": int(ids[-1]) + 1, "n": N}), flush=True)
    if len(rows) != N or [row["pair_id"] for row in rows] != list(range(N)):
        raise QXR1Error("scorer row denominator/order drifted")
    components = aggregate_score(rows)
    result = {
        "schema": SCORE_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": SCORE_AXIS,
        "promotable": False,
        "score_claim": False,
        "verdict_scope": "INSTANCE (exact QXO1 129309-byte archive and current receiver binding)",
        "verdict": "SUB-0.12-CANDIDATE" if components["S"] < 0.12 else "DISTORTION-REFUSED",
        "components": components,
        "archive": binding["archive"],
        "n": N,
        "pair_denominator": N,
        "pixel_denominator": SITES,
        "pose_value_denominator": N * 6,
        "claim": claim,
        "storage_preflight": storage,
        "retained_render_inputs": binding["retained_render_chunks"],
        "retained_scorer_outputs": outputs,
        "pair_rows": atomic_json(output / "PAIR_ROWS.json", rows),
        "field_diff": binding["field_diff"],
        "render_binding": binding["render_binding"],
        "elapsed_seconds": time.time() - started,
        "run_config": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "pointer_moved": False,
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
        "boundaries": [
            "advisory macOS CPU result only",
            "no contest receiver archive/runtime was evaluated",
            "QXO1 sections 7 and 8 are not consumed by the inherited QBT RGB renderer",
        ],
    }
    atomic_json(output / "checkpoints/stage_05_scorer_complete.json", result)
    atomic_json(output / "SCORER_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare", help="scorer-free exact receiver and render-input binding")
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    fire = sub.add_parser("fire-order", help="write the sealed typed MAIN scorer fire order")
    fire.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    scorer = sub.add_parser("score", help="MAIN-only frozen-scorer consumer of prepared payloads")
    scorer.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    scorer.add_argument("--resume-from", type=Path, required=True)
    scorer.add_argument("--scorer-claim-id", required=True)
    scorer.add_argument("--launch-authorized", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output)
    elif args.action == "fire-order":
        result = emit_fire_order(args.output)
    else:
        result = score(
            args.output,
            resume_from=args.resume_from,
            claim_id=args.scorer_claim_id,
            launch_authorized=args.launch_authorized,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
