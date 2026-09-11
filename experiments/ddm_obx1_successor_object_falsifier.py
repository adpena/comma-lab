# SPDX-License-Identifier: MIT
"""n600 falsifier for a receiver-consumed Lane-edge correction on QBT2B r10.

This is a local macOS-CPU advisory instrument, not a contest evaluator.  It
loads the exact retained QBT2B r10 packet, renders the born object, constructs
one counted int8 correction lattice from move 44's shipped Lane-edge geometry,
decodes that lattice before applying it at the camera receiver, and scores both
the born and fused objects with the frozen CPU scorers.  Every materialized
render, correction, scorer output, and coded payload is retained in restartable
chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from experiments import ddm_qbz1_descent_rate_configuration as qbz1
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
from tac.scorer import load_differentiable_scorers

SCHEMA = "ddm_obx1_successor_object_falsifier.v1"
CHUNK_SCHEMA = "ddm_obx1_successor_object_chunk.v1"
N = 600
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
CHANNELS = 3
CHUNK_PAIRS = 20
RATE_DENOMINATOR = 37_545_489
TARGET_SCORE = 0.12
POINTER_ARCHIVE_BYTES = 180_406
POINTER_D_SEG = 0.00010345
POINTER_D_POSE = 4.59e-6
POINTER_DISTORTION = 100.0 * POINTER_D_SEG + math.sqrt(10.0 * POINTER_D_POSE)
QBT2B_B_HAT = 121_928
QBT2B_ARCHIVE_BYTES = 106_714
QBT2B_ARCHIVE_SHA256 = "b26371e50696bdcdafdccbf4c629ef1119ae48aa1ac8765200a6ea2176f91830"
QBT2B_PACKET_SHA256 = "607abebda2708f00daab79aac7bc6839d314096e6ed5693b642525487a1019f7"
QBT2B_CONTAINER_SHA256 = "18d69e4da2024d39ef13e73ef92623ca9857e67cc4f7b551f83d557f9880709d"
POINTER_ARCHIVE_SHA256 = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
POINTER_RAW_SHA256 = "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
POINTER_FIELD_SHA256 = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_obx1_successor_object")
QBT2B_CONTAINER = Path(
    "/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/"
    "governed_n32_r10/stage_05_same_budget_admission/reencoded/stage_05_end/"
    "reencode_payloads.tar"
)
POINTER_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime/archive.zip")
POINTER_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw")
POINTER_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
ACTIVE_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
MINIMUM_FREE_BYTES = 20_000_000_000
LANE_CLASS = 1
CHUNK_MAGIC = b"OBX1Q8B1"
CATALOG_MAGIC = b"OBX1CAT1"
OBJECT_MAGIC = b"OBX1OBJ1"
CHUNK_HEADER = struct.Struct("<8sIIHHB")
OBJECT_HEADER = struct.Struct("<8sQQ")


class OBX1Error(RuntimeError):
    """Fail-closed refusal for OBX1 custody, receiver, or denominator drift."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OBX1Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def require_fact(path: Path, *, digest: str, size: int | None = None) -> dict[str, Any]:
    fact = file_fact(path)
    if fact["sha256"] != digest or (size is not None and fact["bytes"] != size):
        raise OBX1Error(f"frozen input drifted: {path}")
    return fact


def retain_exact(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        fact = file_fact(path)
        if fact["bytes"] != len(payload) or fact["sha256"] != sha256_bytes(payload):
            raise OBX1Error(f"existing retained payload differs; refusing overwrite: {path}")
        return fact
    return qbt1.atomic_bytes(path, payload)


def storage_preflight(output: Path, *, required: int) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise OBX1Error(f"output must be the chartered OBX1 custody root: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free = int(stat.f_bavail * stat.f_frsize)
    if free < required:
        raise OBX1Error(f"Vertigo storage preflight refused: free={free} required={required}")
    return {
        "root": str(output.resolve()),
        "free_bytes": free,
        "required_free_bytes": required,
        "status": "PASS",
        "cleanup": "certify-or-block; this runner never deletes retained OBX1 evidence",
    }


def tar_member_bytes(path: Path, member: str) -> bytes:
    with tarfile.open(path, mode="r") as archive:
        stream = archive.extractfile(member)
        if stream is None:
            raise OBX1Error(f"retained tar member is unreadable: {member}")
        return stream.read()


def model_from_packet(packet: bytes) -> qbt1.QBFLOWTorch:
    decoded = qbf1.decode_packet(packet)
    expected = {
        qbf1.SECTION_CONFIG,
        qbf1.SECTION_MODEL,
        qbf1.SECTION_LATENT_META,
        qbf1.SECTION_LATENTS,
    }
    if set(decoded.sections) != expected:
        raise OBX1Error("receiver-decoded QBF section set differs")
    params = qbf1.decode_model(decoded.sections[qbf1.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(decoded.sections[qbf1.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(decoded.sections[qbf1.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise OBX1Error("receiver-decoded archive does not carry all 600 latent records")
    boundary = np.stack(
        [qbf1.dequantize(records[i][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)) for i in range(N)]
    )
    interior = np.stack(
        [qbf1.dequantize(records[i][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)) for i in range(N)]
    )
    return qbt1.QBFLOWTorch(params, boundary, interior)


def lane_edge_band(field: np.ndarray) -> np.ndarray:
    """Return the Lane-side boundary plus its four-neighbour one-cell annulus."""

    if field.ndim != 3 or field.shape[1:] != (EVAL_H, EVAL_W):
        raise OBX1Error("shipped field geometry differs")
    lane = field == LANE_CLASS
    same_up = np.zeros_like(lane)
    same_down = np.zeros_like(lane)
    same_left = np.zeros_like(lane)
    same_right = np.zeros_like(lane)
    same_up[:, 1:] = lane[:, :-1]
    same_down[:, :-1] = lane[:, 1:]
    same_left[:, :, 1:] = lane[:, :, :-1]
    same_right[:, :, :-1] = lane[:, :, 1:]
    edge = lane & ~(same_up & same_down & same_left & same_right)
    band = edge.copy()
    band[:, 1:] |= edge[:, :-1]
    band[:, :-1] |= edge[:, 1:]
    band[:, :, 1:] |= edge[:, :, :-1]
    band[:, :, :-1] |= edge[:, :, 1:]
    return band


def encode_delta_chunk(delta_q8: np.ndarray, *, start: int) -> bytes:
    if delta_q8.dtype != np.int8 or delta_q8.ndim != 4 or delta_q8.shape[1:] != (CHANNELS, EVAL_H, EVAL_W):
        raise OBX1Error("carrier delta must be int8[N,3,384,512]")
    raw = delta_q8.tobytes(order="C")
    coded = brotli.compress(raw, mode=brotli.MODE_GENERIC, quality=11)
    return CHUNK_HEADER.pack(CHUNK_MAGIC, start, delta_q8.shape[0], EVAL_H, EVAL_W, CHANNELS) + coded


def decode_delta_chunk(payload: bytes) -> tuple[int, np.ndarray]:
    if len(payload) < CHUNK_HEADER.size:
        raise OBX1Error("carrier chunk is truncated")
    magic, start, count, height, width, channels = CHUNK_HEADER.unpack_from(payload)
    if magic != CHUNK_MAGIC or (height, width, channels) != (EVAL_H, EVAL_W, CHANNELS) or count < 1:
        raise OBX1Error("carrier chunk header differs")
    raw = brotli.decompress(payload[CHUNK_HEADER.size :])
    expected = count * channels * height * width
    if len(raw) != expected:
        raise OBX1Error("carrier chunk decoded length differs")
    return start, np.frombuffer(raw, dtype=np.int8).reshape(count, channels, height, width).copy()


def pack_catalog(chunks: Sequence[bytes]) -> bytes:
    if not chunks:
        raise OBX1Error("carrier catalog cannot be empty")
    lengths = struct.pack(f"<{len(chunks)}Q", *(len(chunk) for chunk in chunks))
    return CATALOG_MAGIC + struct.pack("<I", len(chunks)) + lengths + b"".join(chunks)


def unpack_catalog(payload: bytes) -> list[bytes]:
    if len(payload) < 12 or payload[:8] != CATALOG_MAGIC:
        raise OBX1Error("carrier catalog header differs")
    count = struct.unpack_from("<I", payload, 8)[0]
    table_stop = 12 + 8 * count
    if count < 1 or table_stop > len(payload):
        raise OBX1Error("carrier catalog length table differs")
    lengths = struct.unpack_from(f"<{count}Q", payload, 12)
    offset = table_stop
    chunks = []
    for length in lengths:
        stop = offset + length
        if stop > len(payload):
            raise OBX1Error("carrier catalog chunk exceeds payload")
        chunks.append(payload[offset:stop])
        offset = stop
    if offset != len(payload):
        raise OBX1Error("carrier catalog has trailing bytes")
    return chunks


def pack_object(base_packet: bytes, carrier: bytes) -> bytes:
    return OBJECT_HEADER.pack(OBJECT_MAGIC, len(base_packet), len(carrier)) + base_packet + carrier


def unpack_object(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < OBJECT_HEADER.size:
        raise OBX1Error("successor object is truncated")
    magic, base_length, carrier_length = OBJECT_HEADER.unpack_from(payload)
    if magic != OBJECT_MAGIC or OBJECT_HEADER.size + base_length + carrier_length != len(payload):
        raise OBX1Error("successor object framing differs")
    start = OBJECT_HEADER.size
    return payload[start : start + base_length], payload[start + base_length :]


def fuse_camera(born_camera: torch.Tensor, delta_q8: np.ndarray) -> torch.Tensor:
    """Apply the decoded low-resolution correction before the scorer resize."""

    if born_camera.ndim != 5 or born_camera.shape[1:] != (2, CHANNELS, CAMERA_H, CAMERA_W):
        raise OBX1Error("born camera tensor geometry differs")
    delta = torch.from_numpy(delta_q8.astype(np.float32, copy=False))
    camera_delta = F.interpolate(delta, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    fused = born_camera.clone()
    fused[:, 1] = (fused[:, 1] + camera_delta).round().clamp(0.0, 255.0)
    return fused


def assert_active_scorer_claim(claim_id: str) -> dict[str, Any]:
    if not claim_id.startswith("ddm_obx1_"):
        raise OBX1Error("scorer claim must be an OBX1-owned lane id")
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) == 8 and fields[0].startswith("20"):
            rows.append({"timestamp": fields[0], "lane_id": fields[2], "platform": fields[3], "status": fields[6], "raw": line})
    newest_by_lane: dict[str, dict[str, str]] = {}
    for row in rows:
        newest_by_lane.setdefault(row["lane_id"], row)
    own = newest_by_lane.get(claim_id)
    if own is None or own["platform"] != "local_macos_cpu" or not own["status"].startswith("active_"):
        raise OBX1Error("newest OBX1 row must be an active local_macos_cpu scorer claim")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest_by_lane.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if timestamp >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise OBX1Error(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "registry": file_fact(ACTIVE_CLAIMS), "row": own["raw"]}


def stage0(output: Path) -> tuple[dict[str, Any], bytes]:
    storage = storage_preflight(output, required=MINIMUM_FREE_BYTES)
    container_fact = require_fact(QBT2B_CONTAINER, digest=QBT2B_CONTAINER_SHA256)
    base_archive = tar_member_bytes(QBT2B_CONTAINER, "archive.zip")
    base_archive_repeat = tar_member_bytes(QBT2B_CONTAINER, "archive.repeat.zip")
    base_packet = tar_member_bytes(QBT2B_CONTAINER, "packet.qbf")
    base_packet_repeat = tar_member_bytes(QBT2B_CONTAINER, "packet.repeat.qbf")
    if (
        len(base_archive) != QBT2B_ARCHIVE_BYTES
        or sha256_bytes(base_archive) != QBT2B_ARCHIVE_SHA256
        or sha256_bytes(base_packet) != QBT2B_PACKET_SHA256
    ):
        raise OBX1Error("QBT2B r10 archive or packet identity differs")
    if base_archive_repeat != base_archive or base_packet_repeat != base_packet:
        raise OBX1Error("QBT2B r10 deterministic repeat differs")
    if qbf1.read_deterministic_archive(base_archive) != base_packet:
        raise OBX1Error("QBT2B r10 archive receiver does not return the retained packet")
    model_from_packet(base_packet)
    retained = {
        "base_archive": retain_exact(output / "inputs/base.archive.zip", base_archive),
        "base_archive_repeat": retain_exact(output / "inputs/base.archive.repeat.zip", base_archive_repeat),
        "base_packet": retain_exact(output / "inputs/base.packet.qbf", base_packet),
        "base_packet_repeat": retain_exact(output / "inputs/base.packet.repeat.qbf", base_packet_repeat),
    }
    receipt = {
        "schema": "ddm_obx1_stage0.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory custody/receiver check]",
        "score_claim": False,
        "storage_preflight": storage,
        "source_facts": {
            "qbt2b_container": container_fact,
            "pointer_archive": require_fact(POINTER_ARCHIVE, digest=POINTER_ARCHIVE_SHA256, size=POINTER_ARCHIVE_BYTES),
            "pointer_raw": require_fact(POINTER_RAW, digest=POINTER_RAW_SHA256, size=N * 2 * CAMERA_H * CAMERA_W * CHANNELS),
            "pointer_field": require_fact(POINTER_FIELD, digest=POINTER_FIELD_SHA256, size=N * EVAL_H * EVAL_W),
        },
        "retained_inputs": retained,
        "qbt2b_receiver_packet_identity": True,
        "qbt2b_repeat_identity": True,
    }
    qbt1.atomic_json(output / "checkpoints/stage_00_inputs.json", receipt)
    return receipt, base_packet


def score_arrays(
    camera: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
    return (
        pose6.cpu().numpy().astype("<f4"),
        logits.cpu().numpy().astype("<f2"),
        logits.argmax(dim=1).cpu().numpy().astype(np.uint8),
    )


def pair_rows(arrays: Mapping[str, np.ndarray], expected_ids: Sequence[int]) -> list[dict[str, Any]]:
    ids = np.asarray(arrays["pair_ids_i64"], dtype=np.int64)
    if ids.tolist() != list(expected_ids):
        raise OBX1Error("retained chunk pair IDs differ")
    target = np.asarray(arrays["target_argmax_u8"], dtype=np.uint8)
    pose_target = np.asarray(arrays["target_pose6_f32"], dtype=np.float64)
    rows = []
    for index, pair_id in enumerate(ids):
        variants = {}
        for name in ("born", "fused"):
            predicted = np.asarray(arrays[f"{name}_segnet_argmax_u8"], dtype=np.uint8)[index]
            pose = np.asarray(arrays[f"{name}_posenet_pose6_f32"], dtype=np.float64)[index]
            variants[name] = {
                "seg_errors": int((predicted != target[index]).sum()),
                "seg_pixels": int(target[index].size),
                "pose_squared_error_sum": float(np.square(pose - pose_target[index]).sum()),
                "pose_values": 6,
            }
        rows.append({"pair_id": int(pair_id), "variants": variants})
    return rows


def load_chunk(path: Path, expected_ids: Sequence[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "pair_ids_i64",
        "lane_edge_band_u8",
        "carrier_delta_q8",
        "born_camera_pair_u8",
        "fused_camera_pair_u8",
        "born_segnet_logits_f16",
        "born_segnet_argmax_u8",
        "born_posenet_pose6_f32",
        "fused_segnet_logits_f16",
        "fused_segnet_argmax_u8",
        "fused_posenet_pose6_f32",
        "target_argmax_u8",
        "target_pose6_f32",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise OBX1Error(f"retained chunk payload set differs: {path}")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return file_fact(path), pair_rows(arrays, expected_ids)


def realize_chunk(
    output: Path,
    *,
    ids: Sequence[int],
    model: qbt1.QBFLOWTorch,
    pointer_raw: np.memmap,
    field: np.memmap,
    gt: np.ndarray,
    pose_target: np.ndarray,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stem = f"pairs_{ids[0]:04d}_{ids[-1]:04d}"
    payload_path = output / "realized_n600" / f"{stem}.npz"
    carrier_path = output / "carrier_chunks" / f"{stem}.obx1q8"
    carrier_repeat_path = output / "carrier_chunks" / f"{stem}.repeat.obx1q8"
    checkpoint_path = output / "checkpoints" / f"stage_01_{stem}.json"
    if payload_path.exists():
        fact, rows = load_chunk(payload_path, ids)
        carrier = carrier_path.read_bytes()
        start, decoded = decode_delta_chunk(carrier)
        with np.load(payload_path, allow_pickle=False) as payload:
            retained_delta = np.asarray(payload["carrier_delta_q8"], dtype=np.int8)
        if start != ids[0] or not np.array_equal(decoded, retained_delta):
            raise OBX1Error("resumed carrier chunk differs from retained correction")
        if carrier_repeat_path.read_bytes() != carrier:
            raise OBX1Error("resumed carrier deterministic repeat differs")
        checkpoint = {
            "schema": CHUNK_SCHEMA,
            "resumed": True,
            "payload": fact,
            "carrier": file_fact(carrier_path),
            "carrier_repeat": file_fact(carrier_repeat_path),
            "pair_rows": rows,
        }
        if not checkpoint_path.exists():
            qbt1.atomic_json(checkpoint_path, checkpoint)
        return checkpoint, rows

    pair_ids = list(ids)
    with torch.no_grad():
        outputs = model(torch.tensor(pair_ids, dtype=torch.long), height=EVAL_H, width=EVAL_W)
        born_camera = qbt1.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
        pointer_np = np.asarray(pointer_raw[pair_ids], dtype=np.uint8)
        pointer_camera = torch.from_numpy(pointer_np.copy()).permute(0, 1, 4, 2, 3).float()
        born_eval = F.interpolate(born_camera[:, 1], size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
        pointer_eval = F.interpolate(pointer_camera[:, 1], size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
        band = lane_edge_band(np.asarray(field[pair_ids], dtype=np.uint8))
        delta_f32 = (pointer_eval - born_eval) * torch.from_numpy(band[:, None].astype(np.float32))
        unclipped = delta_f32.round()
        delta_q8 = unclipped.clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
        carrier = encode_delta_chunk(delta_q8, start=pair_ids[0])
        start, decoded_delta = decode_delta_chunk(carrier)
        if start != pair_ids[0] or not np.array_equal(decoded_delta, delta_q8):
            raise OBX1Error("fresh carrier chunk failed exact receiver decode")
        carrier_fact = retain_exact(carrier_path, carrier)
        carrier_repeat = encode_delta_chunk(delta_q8, start=pair_ids[0])
        if carrier_repeat != carrier:
            raise OBX1Error("fresh carrier encoder is nondeterministic")
        carrier_repeat_fact = retain_exact(carrier_repeat_path, carrier_repeat)
        fused_camera = fuse_camera(born_camera, decoded_delta)
        born_pose, born_logits, born_argmax = score_arrays(born_camera, posenet=posenet, segnet=segnet)
        fused_pose, fused_logits, fused_argmax = score_arrays(fused_camera, posenet=posenet, segnet=segnet)

    payload_fact = qbt1.atomic_npz(
        payload_path,
        pair_ids_i64=np.asarray(pair_ids, dtype=np.int64),
        lane_edge_band_u8=band.astype(np.uint8),
        carrier_delta_q8=delta_q8,
        born_camera_pair_u8=born_camera.to(torch.uint8).cpu().numpy(),
        fused_camera_pair_u8=fused_camera.to(torch.uint8).cpu().numpy(),
        born_segnet_logits_f16=born_logits,
        born_segnet_argmax_u8=born_argmax,
        born_posenet_pose6_f32=born_pose,
        fused_segnet_logits_f16=fused_logits,
        fused_segnet_argmax_u8=fused_argmax,
        fused_posenet_pose6_f32=fused_pose,
        target_argmax_u8=np.asarray(gt[pair_ids], dtype=np.uint8),
        target_pose6_f32=np.asarray(pose_target[pair_ids], dtype="<f4"),
    )
    _, rows = load_chunk(payload_path, pair_ids)
    checkpoint = {
        "schema": CHUNK_SCHEMA,
        "resumed": False,
        "payload": payload_fact,
        "carrier": carrier_fact,
        "carrier_repeat": carrier_repeat_fact,
        "support_cells": int(band.sum()),
        "nonzero_delta_values": int(np.count_nonzero(delta_q8)),
        "saturated_delta_values": int((np.abs(unclipped.cpu().numpy()) > 127).sum()),
        "pair_rows": rows,
    }
    qbt1.atomic_json(checkpoint_path, checkpoint)
    return checkpoint, rows


def aggregate(pair_rows_all: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pair_rows_all) != N or [int(row["pair_id"]) for row in pair_rows_all] != list(range(N)):
        raise OBX1Error("pair denominator is not exactly 600 ordered rows")
    result = {}
    for name in ("born", "fused"):
        seg_errors = sum(int(row["variants"][name]["seg_errors"]) for row in pair_rows_all)
        seg_pixels = sum(int(row["variants"][name]["seg_pixels"]) for row in pair_rows_all)
        pose_sse = sum(float(row["variants"][name]["pose_squared_error_sum"]) for row in pair_rows_all)
        pose_values = sum(int(row["variants"][name]["pose_values"]) for row in pair_rows_all)
        if seg_pixels != N * EVAL_H * EVAL_W or pose_values != N * 6:
            raise OBX1Error("scorer denominator differs from n600")
        d_seg = seg_errors / seg_pixels
        d_pose = pose_sse / pose_values
        seg_term = 100.0 * d_seg
        pose_term = math.sqrt(10.0 * d_pose)
        result[name] = {
            "seg_errors": seg_errors,
            "seg_pixels": seg_pixels,
            "d_seg": d_seg,
            "pose_squared_error_sum": pose_sse,
            "pose_values": pose_values,
            "d_pose": d_pose,
            "seg_term": seg_term,
            "pose_term": pose_term,
            "distortion": seg_term + pose_term,
        }
    return result


def strict_byte_cap(distortion: float) -> int:
    real_cap = (TARGET_SCORE - distortion) * RATE_DENOMINATOR / 25.0
    return math.ceil(real_cap) - 1


def finalize_object(output: Path, base_packet: bytes, chunk_facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chunks = [Path(str(row["path"])).read_bytes() for row in chunk_facts]
    carrier = pack_catalog(chunks)
    carrier_repeat = pack_catalog(chunks)
    if carrier_repeat != carrier:
        raise OBX1Error("carrier catalog repeat differs")
    carrier_fact = retain_exact(output / "candidate/carrier.obx1cat", carrier)
    carrier_repeat_fact = retain_exact(output / "candidate/carrier.repeat.obx1cat", carrier_repeat)
    object_packet = pack_object(base_packet, carrier)
    object_repeat = pack_object(base_packet, carrier_repeat)
    if object_packet != object_repeat:
        raise OBX1Error("successor object packet repeat differs")
    parsed_base, parsed_carrier = unpack_object(object_packet)
    if parsed_base != base_packet or parsed_carrier != carrier:
        raise OBX1Error("successor object packet receiver differs")
    parsed_chunks = unpack_catalog(parsed_carrier)
    if parsed_chunks != chunks:
        raise OBX1Error("successor carrier catalog receiver differs")
    archive = qbf1.deterministic_archive(object_packet, member_name="0.obx1")
    archive_repeat = qbf1.deterministic_archive(object_repeat, member_name="0.obx1")
    if archive_repeat != archive:
        raise OBX1Error("successor archive repeat differs")
    if qbf1.read_deterministic_archive(archive, member_name="0.obx1") != object_packet:
        raise OBX1Error("successor archive receiver differs")
    return {
        "carrier": carrier_fact,
        "carrier_repeat": carrier_repeat_fact,
        "object_packet": retain_exact(output / "candidate/object.obx1", object_packet),
        "object_packet_repeat": retain_exact(output / "candidate/object.repeat.obx1", object_repeat),
        "archive": retain_exact(output / "candidate/archive.zip", archive),
        "archive_repeat": retain_exact(output / "candidate/archive.repeat.zip", archive_repeat),
        "receiver_base_packet_identity": True,
        "receiver_carrier_identity": True,
        "receiver_chunk_identity": True,
        "archive_repeat_identity": True,
    }


def run(output: Path, *, claim_id: str, launch_authorized: bool, resume_from: Path) -> dict[str, Any]:
    if not launch_authorized:
        raise OBX1Error("n600 scorer realization requires explicit launch authorization")
    if resume_from.resolve() != output.resolve():
        raise OBX1Error("--resume-from must name the exact OBX1 output root")
    claim = assert_active_scorer_claim(claim_id)
    stage0_receipt, base_packet = stage0(output)
    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="OBX1 DALI partition")
    assert_gt_lineage(qbz1.GT_POSE6, required=AUTHORITY_LINEAGE, instrument="OBX1 DALI pose")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, EVAL_H, EVAL_W) or gt.dtype != np.uint8 or pose_target.shape != (N, 6):
        raise OBX1Error("registered scorer-target geometry differs")
    pointer_raw = np.memmap(POINTER_RAW, dtype=np.uint8, mode="r", shape=(N, 2, CAMERA_H, CAMERA_W, CHANNELS))
    field = np.memmap(POINTER_FIELD, dtype=np.uint8, mode="r", shape=(N, EVAL_H, EVAL_W))
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = model_from_packet(base_packet)
    model.eval()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    started = time.time()
    checkpoints = []
    all_rows = []
    for start in range(0, N, CHUNK_PAIRS):
        ids = list(range(start, min(N, start + CHUNK_PAIRS)))
        checkpoint, rows = realize_chunk(
            output,
            ids=ids,
            model=model,
            pointer_raw=pointer_raw,
            field=field,
            gt=gt,
            pose_target=pose_target,
            posenet=posenet,
            segnet=segnet,
        )
        checkpoints.append(checkpoint)
        all_rows.extend(rows)
        print(json.dumps({"realized_pairs": ids[-1] + 1, "n": N, "resumed": checkpoint["resumed"]}), flush=True)
    rows_fact = qbt1.atomic_json(output / "PAIR_ROWS.json", all_rows)
    components = aggregate(all_rows)
    object_receipt = finalize_object(output, base_packet, [row["carrier"] for row in checkpoints])
    candidate_bytes = int(object_receipt["archive"]["bytes"])
    base_rate = 25.0 * QBT2B_ARCHIVE_BYTES / RATE_DENOMINATOR
    candidate_rate = 25.0 * candidate_bytes / RATE_DENOMINATOR
    born_score_physical = components["born"]["distortion"] + base_rate
    fused_score_physical = components["fused"]["distortion"] + candidate_rate
    candidate_cap = strict_byte_cap(components["fused"]["distortion"])
    carrier_bytes = int(object_receipt["carrier"]["bytes"])
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "verdict_scope": "INSTANCE: exact QBT2B r10 plus decoded int8 Lane-edge correction lattice",
        "n": N,
        "pair_denominator": N,
        "pixel_denominator": N * EVAL_H * EVAL_W,
        "pose_value_denominator": N * 6,
        "support_definition": "Lane-side four-neighbour boundary from shipped subset6.u8, dilated by one four-neighbour cell",
        "components": components,
        "rates": {
            "base_archive_bytes_physical": QBT2B_ARCHIVE_BYTES,
            "base_B_hat_selection_estimator": QBT2B_B_HAT,
            "carrier_catalog_bytes": carrier_bytes,
            "candidate_archive_bytes_physical": candidate_bytes,
            "born_score_at_physical_archive_advisory": born_score_physical,
            "fused_score_at_physical_archive_advisory": fused_score_physical,
            "candidate_strict_byte_cap_at_own_distortion": candidate_cap,
            "candidate_bytes_shortfall_vs_own_distortion_cap": candidate_bytes - candidate_cap,
            "candidate_delta_vs_0_12": fused_score_physical - TARGET_SCORE,
            "candidate_delta_vs_pointer_score": fused_score_physical - 0.1372449041713402,
            "cross_counterfactual_B_hat_plus_pointer_distortion": 25.0 * QBT2B_B_HAT / RATE_DENOMINATOR + POINTER_DISTORTION,
            "carrier_distortion_delta": components["fused"]["distortion"] - components["born"]["distortion"],
            "carrier_score_delta_at_physical_bytes": fused_score_physical - born_score_physical,
        },
        "verdict": (
            "DIRECT_LANE_EDGE_Q8_INSTANCE_PASSES_SUB_0_12"
            if fused_score_physical < TARGET_SCORE
            else "DIRECT_LANE_EDGE_Q8_INSTANCE_REFUSED"
        ),
        "source_facts": {
            **stage0_receipt["source_facts"],
            "gt_argmax": require_fact(qbz1.GT_ARGMAX, digest=qbz1.GT_ARGMAX_SHA256),
            "gt_pose6": require_fact(qbz1.GT_POSE6, digest=qbz1.GT_POSE6_SHA256),
            "runner": file_fact(Path(__file__).resolve()),
            "packet_module": file_fact(Path(qbf1.__file__).resolve()),
            "model_module": file_fact(Path(qbt1.__file__).resolve()),
        },
        "run_config": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd().resolve()),
            "seed": 20260911,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "resume_from": str(resume_from.resolve()),
            "chunk_pairs": CHUNK_PAIRS,
        },
        "claim": claim,
        "stage0": stage0_receipt,
        "object": object_receipt,
        "retained_chunks": [row["payload"] for row in checkpoints],
        "retained_carrier_chunks": [row["carrier"] for row in checkpoints],
        "per_pair_rows": rows_fact,
        "all_materialized_payloads_retained": True,
        "elapsed_seconds": time.time() - started,
        "contest_eval_invocations": 0,
        "modal_invocations": 0,
        "pointer_moved": False,
        "boundaries": [
            "local macOS CPU frozen-scorer advisory; not contest CPU/CUDA authority",
            "the archive is an exact counted research container but no public contest runtime is claimed",
            "the correction copies pointer-derived RGB differences and is an optimistic oracle-like falsifier",
            "the refusal, if any, closes only this direct int8 Lane-edge lattice instance",
        ],
    }
    qbt1.atomic_json(output / "checkpoints/stage_01_complete.json", result)
    qbt1.atomic_json(output / "RESULT.json", result)
    qbt1.atomic_json(
        output / "checkpoints/stage_02_verdict.json",
        {"schema": "ddm_obx1_stage2_verdict.v1", "verdict": result["verdict"], "rates": result["rates"]},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    realize = subparsers.add_parser("realize")
    realize.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    realize.add_argument("--resume-from", type=Path, required=True)
    realize.add_argument("--claim-id", required=True)
    realize.add_argument("--launch-authorized", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        receipt, _ = stage0(args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    result = run(
        args.output,
        claim_id=args.claim_id,
        launch_authorized=args.launch_authorized,
        resume_from=args.resume_from,
    )
    print(json.dumps({"verdict": result["verdict"], "rates": result["rates"], "components": result["components"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
