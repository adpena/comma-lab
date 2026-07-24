#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure and persist the DDM RD1 restricted λ-continuation frontier.

This is a local-only, false-authority measurement tool.  It replays the exact
C1 receiver output through the pinned CPU scorers at four Torch threads,
reaggregates all preserved Menu1 n600 checkpoints, rehashes every V19C n600
candidate archive, and then solves the resulting finite measured domain.

The ``--refine-dimensions-only`` mode never replays a scorer or restarts a
lambda point. It types the completed adjacent duals by stratum, scorer
visibility, and G4 temporal class, then fails closed wherever the existing
endpoints lack a joint byte home or effective-quantum measurement.

No contest evaluator, provider dispatch, training, or frontier mutation is
reachable from this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_lambda_continuation_frontier import (  # noqa: E402
    EVIDENCE_AXIS,
    G4_TEMPORAL_CLASSES,
    CodedStream,
    MeasuredDescription,
    canonical_json_bytes,
    continuation_rows,
    discrete_dual_rows,
    geometric_curvature_ladder,
    lower_supported_hull,
    metric_active_continuation_geometry_report,
    normalized_knee,
    pareto_nondominated,
    publish_immutable_json,
    realized_distortion,
    second_order_metric_geometry_addendum_report,
    sha256_bytes,
    typed_dimension_dual_report,
)

CAMERA_HW = (874, 1164)
SEG_HW = (384, 512)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
DEFAULT_CONFIG = REPO / ".omx/research/configs/ddm_rd1_lambda_continuation_frontier_20260724.json"
DEFAULT_DIMENSION_CONFIG = REPO / ".omx/research/configs/ddm_rd1_dimension_duals_20260724.json"


class RD1MeasurementError(RuntimeError):
    """Raised when a measurement or custody invariant fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RD1MeasurementError(f"JSON root must be an object: {path}")
    return value


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _bound_json(path_value: str, expected_sha256: str, label: str) -> dict[str, Any]:
    path = _repo_path(path_value)
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RD1MeasurementError(f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}")
    return _read_json(path)


def _description_from_receipt(row: Mapping[str, Any]) -> MeasuredDescription:
    return MeasuredDescription(
        candidate_id=str(row["candidate_id"]),
        counted_bytes=int(row["counted_bytes"]),
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
        coded_streams=tuple(
            CodedStream(
                stream_id=str(stream["stream_id"]),
                stratum=str(stream["stratum"]),
                factor_kind=str(stream["factor_kind"]),
                custody_role=str(stream["custody_role"]),
                counted_bytes=int(stream["counted_bytes"]),
                sha256=str(stream["sha256"]),
                codec=str(stream["codec"]),
                source_path=str(stream["source_path"]),
            )
            for stream in row["coded_streams"]
        ),
        source_artifact=str(row["source_artifact"]),
        source_sha256=str(row["source_sha256"]),
        receiver_closure=str(row["receiver_closure"]),
        pool_id=str(row["pool_id"]),
        pair_count=int(row["pair_count"]),
        evidence_axis=str(row["evidence_axis"]),
        score_claim=bool(row["score_claim"]),
        own_stored_problem=bool(row["own_stored_problem"]),
        donor_conditioned=bool(row["donor_conditioned"]),
        per_class=row.get("per_class"),
        metadata=row.get("metadata"),
    )


def _file_hash_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    checkpoint_root: Path,
    label: str,
) -> dict[str, Any]:
    """Hash a large immutable input once, then resume by size+mtime identity."""

    stat = path.stat()
    safe = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    receipt_path = checkpoint_root / "00_input_hashes" / f"{safe}.json"
    if receipt_path.exists():
        row = _read_json(receipt_path)
        if (
            row.get("path") == str(path)
            and row.get("bytes") == stat.st_size
            and row.get("mtime_ns") == stat.st_mtime_ns
            and row.get("sha256") == expected_sha256
        ):
            return row
    print(f"[RD1] hash {label}: {path}", flush=True)
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RD1MeasurementError(f"{label} SHA-256 differs: expected {expected_sha256}, observed {observed}")
    row = {
        "schema": "ddm_rd1_input_hash.v1",
        "label": label,
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": observed,
    }
    publish_immutable_json(receipt_path, row)
    return row


def _storage_preflight(checkpoint_root: Path) -> dict[str, Any]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not str(checkpoint_root).startswith("/Volumes/VertigoDataTier/pact/"):
        raise RD1MeasurementError("checkpoint root must use the primary SSD tier")
    required = 1 << 30
    free = shutil.disk_usage(checkpoint_root).free
    if free < required:
        raise RD1MeasurementError("SSD storage preflight failed")
    return {
        "schema": "ddm_rd1_storage_preflight.v1",
        "tier": "/Volumes/VertigoDataTier/pact",
        "checkpoint_root": str(checkpoint_root),
        "required_free_bytes": required,
        "observed_free_bytes_at_least": required,
        "observation_policy": (
            "checked afresh before each execution; exact live free-space bytes "
            "are excluded from immutable receipts because they are volatile"
        ),
        "status": "PASS",
        "cleanup_policy": (
            "preserve immutable per-lambda and per-scorer-batch checkpoints; "
            "no scratch decoder output is created by RD1"
        ),
    }


def _zip_member_data_offset(blob: bytes, info: zipfile.ZipInfo) -> int:
    offset = info.header_offset
    if blob[offset : offset + 4] != b"PK\x03\x04":
        raise RD1MeasurementError("ZIP local header signature differs")
    name_len, extra_len = struct.unpack_from("<HH", blob, offset + 26)
    return offset + 30 + name_len + extra_len


def _factor_for_path(path: str) -> str:
    lowered = path.lower()
    if (
        "manifest" in lowered
        or "/structure/" in lowered
        or lowered.startswith("structure/")
        or lowered.endswith(".g1s")
        or lowered.endswith("/zip_framing")
    ):
        return "skeleton"
    return "fiber"


def _role_for_path(path: str) -> str:
    lowered = path.lower()
    if "/render/" in lowered or lowered.startswith("render/"):
        return "solve_exception"
    return "stored_problem"


def _split_small_zip(
    blob: bytes,
    *,
    archive_path: Path,
    prefix: str = "outer",
    depth: int = 0,
) -> tuple[CodedStream, ...]:
    """Recursively split a small stored ZIP into exact, nonoverlapping byte homes."""

    if depth > 12:
        raise RD1MeasurementError("nested description ZIP depth exceeded")
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise RD1MeasurementError("description archive is not a valid ZIP") from exc
    infos = archive.infolist()
    spans: list[tuple[int, int]] = []
    streams: list[CodedStream] = []
    for info in infos:
        start = _zip_member_data_offset(blob, info)
        stop = start + info.compress_size
        spans.append((start, stop))
        member_payload = blob[start:stop]
        member_path = f"{prefix}/{info.filename}"
        if (
            info.compress_type == zipfile.ZIP_STORED
            and info.filename.endswith(".zip")
            and member_payload.startswith(b"PK")
        ):
            streams.extend(
                _split_small_zip(
                    member_payload,
                    archive_path=archive_path,
                    prefix=member_path,
                    depth=depth + 1,
                )
            )
            continue
        if not member_payload:
            raise RD1MeasurementError("zero-byte coded stream is not admitted")
        streams.append(
            CodedStream(
                stream_id=member_path,
                stratum=member_path.split("/", 2)[1],
                factor_kind=_factor_for_path(member_path),  # type: ignore[arg-type]
                custody_role=_role_for_path(member_path),  # type: ignore[arg-type]
                counted_bytes=len(member_payload),
                sha256=sha256_bytes(member_payload),
                codec=(
                    "ZIP_STORED" if info.compress_type == zipfile.ZIP_STORED else f"ZIP_METHOD_{info.compress_type}"
                ),
                source_path=f"{archive_path}::{member_path}",
            )
        )
    framing = bytearray()
    cursor = 0
    for start, stop in sorted(spans):
        framing.extend(blob[cursor:start])
        cursor = stop
    framing.extend(blob[cursor:])
    if not framing:
        raise RD1MeasurementError("ZIP framing byte home is empty")
    framing_id = f"{prefix}/zip_framing"
    streams.append(
        CodedStream(
            stream_id=framing_id,
            stratum="container",
            factor_kind="skeleton",
            custody_role="stored_problem",
            counted_bytes=len(framing),
            sha256=sha256_bytes(bytes(framing)),
            codec="ZIP_FRAMING",
            source_path=f"{archive_path}::{framing_id}",
        )
    )
    if sum(stream.counted_bytes for stream in streams) != len(blob):
        raise RD1MeasurementError("recursive ZIP streams do not partition archive bytes")
    return tuple(sorted(streams, key=lambda stream: stream.stream_id))


def _split_c1_archive(
    path: Path,
    *,
    packet_sha256: str,
    packet_bytes: int,
) -> tuple[CodedStream, ...]:
    """Split the 409 MB C1 ZIP without materializing its packet in memory."""

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "0.bin":
            raise RD1MeasurementError("C1 archive member layout differs")
        info = infos[0]
        if (
            info.compress_type != zipfile.ZIP_STORED
            or info.file_size != packet_bytes
            or info.compress_size != packet_bytes
        ):
            raise RD1MeasurementError("C1 packet byte home differs")
        with path.open("rb") as handle:
            local_header = handle.read(info.header_offset + 30)
            if local_header[info.header_offset : info.header_offset + 4] != b"PK\x03\x04":
                raise RD1MeasurementError("C1 local ZIP header differs")
            name_len, extra_len = struct.unpack_from("<HH", local_header, info.header_offset + 26)
            data_offset = info.header_offset + 30 + name_len + extra_len
            handle.seek(0)
            prefix = handle.read(data_offset)
            handle.seek(data_offset + packet_bytes)
            suffix = handle.read()
    framing = prefix + suffix
    if len(framing) + packet_bytes != path.stat().st_size:
        raise RD1MeasurementError("C1 byte homes do not partition archive")
    return (
        CodedStream(
            stream_id="c1/plane_solution_packet",
            stratum="full_rank_uint8_solution",
            factor_kind="fiber",
            custody_role="stored_problem",
            counted_bytes=packet_bytes,
            sha256=packet_sha256,
            codec="C1_TWO_PLANE_PACKET",
            source_path=f"{path}::0.bin",
        ),
        CodedStream(
            stream_id="c1/zip_framing",
            stratum="container",
            factor_kind="skeleton",
            custody_role="stored_problem",
            counted_bytes=len(framing),
            sha256=sha256_bytes(framing),
            codec="ZIP_FRAMING",
            source_path=f"{path}::zip_framing",
        ),
    )


def _aggregate_measurement_batches(
    root: Path,
    candidate_id: str,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    stage = root / "02_measurements" / candidate_id
    paths = sorted(stage.glob("batch_*.json"))
    if len(paths) != 38:
        raise RD1MeasurementError(f"{candidate_id} must have 38 preserved n600 batch checkpoints")
    rows = [_read_json(path) for path in paths]
    if any(row.get("candidate_id") != candidate_id for row in rows):
        raise RD1MeasurementError("Menu1 batch candidate identity differs")
    if config_sha256 is not None and any(row.get("typed_config_sha256") != config_sha256 for row in rows):
        raise RD1MeasurementError("Menu1 batch config identity differs")
    ranges = [tuple(row["pair_range"]) for row in rows]
    expected = [(start, min(start + 16, 600)) for start in range(0, 600, 16)]
    if ranges != expected:
        raise RD1MeasurementError("Menu1 batch ranges are not complete and ordered")
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    per_class = {}
    for class_name in CLASS_NAMES.values():
        class_errors = sum(int(row["per_class"][class_name]["errors"]) for row in rows)
        class_sites = sum(int(row["per_class"][class_name]["sites"]) for row in rows)
        per_class[class_name] = {
            "errors": class_errors,
            "sites": class_sites,
            "d_seg": class_errors / class_sites,
        }
    return {
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "d_pose": pose_sse / pose_coordinates,
        "per_class": per_class,
        "batch_count": len(rows),
        "batch_digest_chain_sha256": sha256_bytes(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in rows).encode("ascii")
        ),
    }


def _load_models(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    upstream = Path(str(config["upstream_root"])).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    modules_path = upstream / "modules.py"
    seg_path = upstream / "models" / "segnet.safetensors"
    pose_path = upstream / "models" / "posenet.safetensors"
    for path, key, label in (
        (modules_path, "modules_sha256", "modules.py"),
        (seg_path, "segnet_weights_sha256", "SegNet weights"),
        (pose_path, "posenet_weights_sha256", "PoseNet weights"),
    ):
        observed = _sha256_file(path)
        if observed != config[key]:
            raise RD1MeasurementError(f"{label} SHA-256 differs")
    spec = importlib.util.spec_from_file_location("ddm_rd1_upstream_modules", modules_path)
    if spec is None or spec.loader is None:
        raise RD1MeasurementError("cannot import frozen scorer modules")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)
    threads = int(config["scorer_threads"])
    if threads != 4:
        raise RD1MeasurementError("E3 deterministic scorer law requires four threads")
    torch.set_num_threads(threads)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    segnet = modules.SegNet().eval().cpu()
    posenet = modules.PoseNet().eval().cpu()
    segnet.load_state_dict(load_file(str(seg_path), device="cpu"), strict=True)
    posenet.load_state_dict(load_file(str(pose_path), device="cpu"), strict=True)
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    custody = {
        "modules_path": str(modules_path),
        "modules_sha256": config["modules_sha256"],
        "segnet_weights_path": str(seg_path),
        "segnet_weights_sha256": config["segnet_weights_sha256"],
        "posenet_weights_path": str(pose_path),
        "posenet_weights_sha256": config["posenet_weights_sha256"],
        "device": "cpu",
        "threads": threads,
        "batch_size": int(config["scorer_batch_size"]),
        "deterministic_algorithms": True,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return segnet, posenet, custody


def _forward(segnet: Any, posenet: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    camera = np.asarray(camera_pairs)
    if camera.dtype != np.uint8 or camera.ndim != 5 or camera.shape[1:] != (2, *CAMERA_HW, 3):
        raise RD1MeasurementError("C1 scorer camera geometry differs")
    owned = np.array(camera, dtype=np.uint8, order="C", copy=True)
    tensor = torch.from_numpy(owned).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        cells = segnet(segnet.preprocess_input(tensor)).argmax(dim=1).cpu().numpy().astype(np.uint8)
        pose_output = posenet(posenet.preprocess_input(tensor))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def _verify_c1_scorers(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    checkpoint_root: Path,
) -> dict[str, Any]:
    stage = checkpoint_root / "01_c1_n600_frozen_scorer"
    aggregate_path = stage / "aggregate.json"
    if aggregate_path.exists():
        aggregate = _read_json(aggregate_path)
        if aggregate.get("typed_config_sha256") != config_sha256:
            raise RD1MeasurementError("C1 aggregate resume config differs")
        return aggregate

    raw_path = Path(str(config["c1_raw_path"]))
    expected_raw_bytes = 2 * 600 * CAMERA_HW[0] * CAMERA_HW[1] * 3
    if raw_path.stat().st_size != expected_raw_bytes:
        raise RD1MeasurementError("C1 raw byte count differs")
    raw = np.memmap(
        raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * 600, *CAMERA_HW, 3),
    )
    cache = Path(str(config["target_cache_path"]))
    labels = open_stored_npy_memmap(cache, "lstars")
    poses = open_stored_npy_memmap(cache, "gt_poses")
    if labels.shape != (600, *SEG_HW) or poses.shape != (600, 6):
        raise RD1MeasurementError("target cache n600 geometry differs")
    segnet = posenet = custody = None
    batch_rows: list[dict[str, Any]] = []
    batch_size = int(config["scorer_batch_size"])
    for start in range(0, 600, batch_size):
        stop = min(start + batch_size, 600)
        path = stage / f"batch_{start:04d}_{stop:04d}.json"
        if path.exists():
            row = _read_json(path)
            if row.get("typed_config_sha256") != config_sha256 or row.get("pair_range") != [start, stop]:
                raise RD1MeasurementError("C1 scorer batch resume identity differs")
            batch_rows.append(row)
            continue
        if segnet is None or posenet is None:
            segnet, posenet, custody = _load_models(config)
        print(f"[RD1] C1 exact-anchor scorer batch {start:04d}:{stop:04d}", flush=True)
        camera = np.asarray(raw[2 * start : 2 * stop]).reshape(stop - start, 2, *CAMERA_HW, 3)
        cells, pose6 = _forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose = _forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(pose6, replay_pose):
                raise RD1MeasurementError("C1 first-batch deterministic replay differs")
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        per_class = {}
        for class_id, class_name in CLASS_NAMES.items():
            mask = target == class_id
            per_class[class_name] = {
                "errors": int(np.count_nonzero((cells != target) & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        row = {
            "schema": "ddm_rd1_c1_scorer_batch.v1",
            "typed_config_sha256": config_sha256,
            "pair_range": [start, stop],
            "errors": int(np.count_nonzero(cells != target)),
            "sites": int(cells.size),
            "pose_squared_error_sum": float(np.square(pose6 - target_pose).sum(dtype=np.float64)),
            "pose_coordinates": int(pose6.size),
            "per_class": per_class,
            "camera_sha256": sha256_bytes(np.ascontiguousarray(camera).tobytes()),
            "cells_sha256": sha256_bytes(cells.tobytes()),
            "pose6_sha256": sha256_bytes(pose6.tobytes()),
            "threads": 4,
            "batch_size": 16,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        publish_immutable_json(path, row)
        batch_rows.append(row)
    errors = sum(int(row["errors"]) for row in batch_rows)
    sites = sum(int(row["sites"]) for row in batch_rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in batch_rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batch_rows)
    per_class = {}
    for class_name in CLASS_NAMES.values():
        class_errors = sum(int(row["per_class"][class_name]["errors"]) for row in batch_rows)
        class_sites = sum(int(row["per_class"][class_name]["sites"]) for row in batch_rows)
        per_class[class_name] = {
            "errors": class_errors,
            "sites": class_sites,
            "d_seg": class_errors / class_sites,
        }
    if custody is None:
        custody = {
            "resume": "all batch checkpoints already present",
            "threads": 4,
            "batch_size": 16,
            "evidence_axis": EVIDENCE_AXIS,
        }
    aggregate = {
        "schema": "ddm_rd1_c1_n600_frozen_scorer_aggregate.v1",
        "typed_config_sha256": config_sha256,
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "d_pose": pose_sse / pose_coordinates,
        "D_realized": realized_distortion(errors / sites, pose_sse / pose_coordinates),
        "per_class": per_class,
        "batch_count": len(batch_rows),
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": sha256_bytes(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in batch_rows).encode("ascii")
        ),
        "scorer_custody": custody,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    publish_immutable_json(aggregate_path, aggregate)
    return aggregate


def _load_v19_candidates(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> list[MeasuredDescription]:
    root = _repo_path(str(config["v19c_candidate_archive_root"]))
    decision_root = root.parent / "stage_checkpoints" / "02_n600_decisions"
    candidates: list[MeasuredDescription] = []
    rows = receipt["curve"]["n600_per_admitted_move"]
    if len(rows) != 104:
        raise RD1MeasurementError("V19C n600 curve length differs")
    accepted_decisions = []
    for decision_path in sorted(decision_root.glob("candidate_*.json")):
        decision = _read_json(decision_path)
        if decision.get("accepted") is True:
            accepted_decisions.append((decision_path, decision))
    if len(accepted_decisions) != len(rows):
        raise RD1MeasurementError("V19C accepted decision count differs from curve")
    for row, (decision_path, decision) in zip(rows, accepted_decisions, strict=True):
        admission_index = int(row["admission_index"])
        if admission_index != len(candidates) or decision["proposal"]["candidate_id"] != row["proposal_id"]:
            raise RD1MeasurementError("V19C accepted decision ordering differs")
        path = _repo_path(str(decision["archive"]["path"]))
        if path.parent != root:
            raise RD1MeasurementError("V19C decision resolved outside candidate archive root")
        blob = path.read_bytes()
        source_sha = sha256_bytes(blob)
        if (
            len(blob) != int(row["archive_bytes"])
            or len(blob) != int(decision["archive"]["bytes"])
            or source_sha != decision["archive"]["sha256"]
        ):
            raise RD1MeasurementError("V19C candidate archive bytes differ from curve")
        streams = _split_small_zip(blob, archive_path=path)
        candidates.append(
            MeasuredDescription(
                candidate_id=f"v19c_admit_{admission_index:04d}",
                counted_bytes=len(blob),
                d_seg=float(row["d_seg"]),
                d_pose=float(row["d_pose"]),
                coded_streams=streams,
                source_artifact=str(path.relative_to(REPO)),
                source_sha256=source_sha,
                receiver_closure="archive_receiver_closed",
                metadata={
                    "source_family": "V19C",
                    "family": row["family"],
                    "proposal_id": row["proposal_id"],
                    "admission_index": admission_index,
                    "decision_checkpoint": str(decision_path.relative_to(REPO)),
                    "trial_candidate_index": int(decision_path.stem.split("_")[-1]),
                    "bucket_transition": row["bucket_transition"],
                    "mechanism_bucket": "V19C_RESTRICTED_LATTICE_CORRECTION",
                },
            )
        )
    return candidates


def _menu_payload_components(
    row_id: str,
    receipt: Mapping[str, Any],
) -> tuple[tuple[Path, str, str], ...]:
    global_rows = receipt["global_temporal_ladder_payloads"]
    statistics = receipt["local_statistics_payload"]
    targeted = receipt["targeted_cluster_payload"]
    mapping: dict[str, tuple[tuple[str, str, str], ...]] = {
        "scalar_gain_bias_12b_frame1": (
            (
                global_rows["scalar"]["payload_path"],
                global_rows["scalar"]["payload_sha256"],
                "global_scalar_affine",
            ),
        ),
        "temporal_affine_16knot_frame1": (
            (
                global_rows["temporal"]["payload_path"],
                global_rows["temporal"]["payload_sha256"],
                "temporal_affine",
            ),
        ),
        "local_statistics_16band_frame1": (
            (
                statistics["payload_path"],
                statistics["payload_sha256"],
                "local_statistics",
            ),
        ),
        "statistics_hard_analytic_composed_frame1": (
            (
                statistics["payload_path"],
                statistics["payload_sha256"],
                "local_statistics",
            ),
        ),
        "top_sn1_cluster_targeted_prototype_frame1": (
            (
                statistics["payload_path"],
                statistics["payload_sha256"],
                "local_statistics",
            ),
            (
                targeted["payload_path"],
                targeted["payload_sha256"],
                "targeted_cluster_mask",
            ),
        ),
    }
    if row_id not in mapping:
        raise RD1MeasurementError(f"unknown Menu1 row: {row_id}")
    result = []
    for raw_path, expected_sha, label in mapping[row_id]:
        path = Path(raw_path)
        observed = _sha256_file(path)
        if observed != expected_sha:
            raise RD1MeasurementError(f"Menu1 {label} payload SHA-256 differs")
        result.append((path, observed, label))
    return tuple(result)


def _load_menu_candidates(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    v19_current: MeasuredDescription,
) -> tuple[list[MeasuredDescription], dict[str, Any]]:
    root = Path(str(config["menu1_checkpoint_root"]))
    menu_config_sha = receipt["typed_config_sha256"]
    base_aggregate = _aggregate_measurement_batches(root, "v19c_base", config_sha256=menu_config_sha)
    if not math.isclose(base_aggregate["d_seg"], v19_current.d_seg, abs_tol=1e-15) or not math.isclose(
        base_aggregate["d_pose"], v19_current.d_pose, abs_tol=1e-12
    ):
        raise RD1MeasurementError("Menu1 base reaggregation differs from V19C current")
    candidates: list[MeasuredDescription] = []
    for row in receipt["measured_menu_rows"]:
        row_id = str(row["row_id"])
        aggregate = _aggregate_measurement_batches(root, row_id, config_sha256=menu_config_sha)
        if not math.isclose(aggregate["d_seg"], float(row["d_seg"]), abs_tol=1e-15) or not math.isclose(
            aggregate["d_pose"], float(row["d_pose"]), abs_tol=1e-12
        ):
            raise RD1MeasurementError(f"Menu1 {row_id} reaggregation differs")
        components = _menu_payload_components(row_id, receipt)
        streams = list(v19_current.coded_streams)
        for path, payload_sha, label in components:
            streams.append(
                CodedStream(
                    stream_id=f"menu1/{label}",
                    stratum=str(row["cluster_id"]),
                    factor_kind="fiber",
                    custody_role="solve_exception",
                    counted_bytes=path.stat().st_size,
                    sha256=payload_sha,
                    codec="DDM_TYPED_BINARY",
                    source_path=str(path),
                )
            )
        counted = sum(stream.counted_bytes for stream in streams)
        if counted != int(row["archive_bytes"]):
            raise RD1MeasurementError(f"Menu1 {row_id} typed byte partition differs")
        candidates.append(
            MeasuredDescription(
                candidate_id=row_id,
                counted_bytes=counted,
                d_seg=aggregate["d_seg"],
                d_pose=aggregate["d_pose"],
                coded_streams=tuple(streams),
                source_artifact=str(config["menu1_receipt_path"]),
                source_sha256=str(config["menu1_receipt_sha256"]),
                receiver_closure="measurement_harness_receiver_closed",
                per_class=aggregate["per_class"],
                metadata={
                    "source_family": "Menu1",
                    "mechanism_bucket": row["mechanism_bucket"],
                    "composition_pool_id": row["composition_pool_id"],
                    "prior_menu_admitted": row["admitted"],
                    "prior_menu_admission_reason": row["admission_reason"],
                    "measurement_status": row["measurement_status"],
                    "transform": next(item["transform"] for item in receipt["curve"] if item["candidate_id"] == row_id)
                    if any(item["candidate_id"] == row_id for item in receipt["curve"])
                    else row_id,
                    "bundle_components": [str(path) for path, _sha, _label in components],
                },
            )
        )
    return candidates, base_aggregate


def _replace_per_class(
    candidate: MeasuredDescription,
    per_class: Mapping[str, Mapping[str, float | int]],
) -> MeasuredDescription:
    return MeasuredDescription(
        candidate_id=candidate.candidate_id,
        counted_bytes=candidate.counted_bytes,
        d_seg=candidate.d_seg,
        d_pose=candidate.d_pose,
        coded_streams=candidate.coded_streams,
        source_artifact=candidate.source_artifact,
        source_sha256=candidate.source_sha256,
        receiver_closure=candidate.receiver_closure,
        pool_id=candidate.pool_id,
        pair_count=candidate.pair_count,
        evidence_axis=candidate.evidence_axis,
        score_claim=candidate.score_claim,
        own_stored_problem=candidate.own_stored_problem,
        donor_conditioned=candidate.donor_conditioned,
        per_class=per_class,
        metadata=candidate.metadata,
    )


def _build_knee_bundle(
    knee: MeasuredDescription,
    *,
    output_root: Path,
    config_sha256: str,
) -> dict[str, Any]:
    bundle_path = output_root / f"knee_{knee.candidate_id}.full-description.zip"
    component_paths: list[Path]
    if knee.metadata and knee.metadata.get("source_family") == "Menu1":
        component_paths = [_repo_path(knee.source_artifact)]
        # The receipt is provenance; counted components are the V19C base plus
        # the exact Menu1 payloads.
        base = next(
            Path(stream.source_path.split("::", 1)[0])
            for stream in knee.coded_streams
            if stream.source_path.endswith("::outer/manifest.json")
        )
        component_paths = [base]
        component_paths.extend(Path(path) for path in knee.metadata["bundle_components"])
    else:
        component_paths = [_repo_path(knee.source_artifact)]
    unique: list[Path] = []
    seen = set()
    for path in component_paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    manifest = {
        "schema": "ddm_rd1_knee_full_description_bundle.v1",
        "candidate_id": knee.candidate_id,
        "typed_config_sha256": config_sha256,
        "logical_counted_bytes": knee.counted_bytes,
        "description_root_sha256": knee.description_root_sha256,
        "receiver_closure": knee.receiver_closure,
        "bundle_is_custody_container_not_counted_archive": True,
        "components": [
            {
                "bundle_name": f"components/{index:02d}_{path.name}",
                "source_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for index, path in enumerate(unique)
        ],
        "receiver_recipe": {
            "transform": (knee.metadata or {}).get("transform"),
            "source_module": "tools/measure_ddm_menu1_realized_flip_menu.py",
            "source_module_sha256": _sha256_file(REPO / "tools/measure_ddm_menu1_realized_flip_menu.py"),
            "authority": "measurement_harness_receiver_closed_not_contest_archive_closed",
        },
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in [
            ("manifest.json", canonical_json_bytes(manifest) + b"\n"),
            *[
                (
                    component["bundle_name"],
                    path.read_bytes(),
                )
                for component, path in zip(manifest["components"], unique, strict=True)
            ],
        ]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    payload = memory.getvalue()
    if bundle_path.exists() and bundle_path.read_bytes() != payload:
        raise RD1MeasurementError("immutable knee description bundle differs")
    if not bundle_path.exists():
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, bundle_path)
    return {
        "path": str(bundle_path.relative_to(REPO)),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "logical_counted_bytes": knee.counted_bytes,
        "description_root_sha256": knee.description_root_sha256,
        "bundle_is_custody_container_not_counted_archive": True,
        "receiver_closure": knee.receiver_closure,
    }


def _dr2b_crosscheck(receipt: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for source in receipt["g2_costate_rows"]["rows"]:
        rebase = source["n600_rebase"]
        rows.append(
            {
                "probe_id": source["probe_id"],
                "purpose": source["purpose"],
                "costate_rank": source.get("costate_rank"),
                "delta_bytes": rebase["delta_bytes"],
                "delta_d_seg": rebase["delta_d_seg"],
                "delta_d_pose": rebase["delta_d_pose"],
                "delta_D_realized": rebase["distortion_delta"],
                "joint_delta_with_rate": rebase["joint_delta"],
                "reverse_waterfill_admissible": rebase["reverse_waterfill_admissible"],
                "verdict_scope": source["verdict_scope"],
            }
        )
    return {
        "source_receipt_sha256": receipt["typed_config_sha256"],
        "rows": rows,
        "derivation_vs_sweep": (
            "DIRECTIONAL_ONLY: two DR2b semantic coordinate instances free bytes "
            "without distortion debt, agreeing that some local tolerances are free. "
            "Numerical dual transfer is blocked because DR2b explicitly lacks the "
            "SDWL1/E2 coordinate crosswalk and RD1 spans different descriptions."
        ),
        "coordinate_crosswalk_status": receipt["u1_lossy_tolerance_ladder"]["sdwl1_crosswalk"]["status"],
        "verdict_scope": (
            "INSTANCE cross-check of five E2 coordinates against a restricted "
            "description-level continuation; no formulation-wide multiplier transfer"
        ),
    }


def _train_decision_solve_table(
    duals: Sequence[Mapping[str, Any]],
    *,
    dr2b: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for row in duals:
        lambda_value = float(row["lambda_bytes_per_D"])
        if lambda_value < 1_000.0:
            disposition = "SOLVE_CHEAPLY_IN_RESTRICTED_MEASURED_POOL"
        elif lambda_value < 1_000_000.0:
            disposition = "SOLVE_EXPENSIVE_MEASURE_NEXT_NEIGHBOR"
        else:
            disposition = "TRAIN_EARNER_CANDIDATE_RATE_GAP"
        result.append(
            {
                "bucket": row["constraint_group"],
                "SOLVE": disposition,
                "lambda_bytes_per_D": lambda_value,
                "left_candidate_id": row["left_candidate_id"],
                "right_candidate_id": row["right_candidate_id"],
                "per_class_d_seg": row["per_class_d_seg"],
                "scope": "restricted measured n600 description pool",
            }
        )
    result.append(
        {
            "bucket": "DR2B_E2_LOCAL_TOLERANCE",
            "SOLVE": "TWO_FREE_INSTANCE_ROWS; FORMULATION_CROSSWALK_BLOCKED",
            "lambda_bytes_per_D": None,
            "left_candidate_id": None,
            "right_candidate_id": None,
            "per_class_d_seg": [],
            "scope": dr2b["verdict_scope"],
        }
    )
    return result


def _effective_quantum_tolerance_report(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose measured coordinate sensitivities without inventing uint8 quanta."""

    component_evidence: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for source in receipt["g2_costate_rows"]["rows"]:
        perturbation = source["perturbation"]
        rebase = source["n600_rebase"]
        serialized_steps = abs(int(perturbation["delta"]))
        if serialized_steps <= 0:
            raise RD1MeasurementError("DR2b serialized perturbation step must be nonzero")
        margin = source.get("fisher_margin")
        stratum = (
            CLASS_NAMES[int(margin["top1_class"])]
            if margin is not None
            else "GLOBAL_CHART"
        )
        visibility_terms = []
        if float(rebase["seg_term"]) != 0.0:
            visibility_terms.append(("seg-visible", float(rebase["seg_term"])))
        if float(rebase["pose_term"]) != 0.0:
            visibility_terms.append(("pose-visible", float(rebase["pose_term"])))
        if not visibility_terms:
            visibility_terms.append(("ker(A)-invisible", 0.0))
        for scorer_visibility, delta_D in visibility_terms:
            component = {
                "probe_id": source["probe_id"],
                "stratum": stratum,
                "scorer_visibility": scorer_visibility,
                "g4_temporal_class": "UNRESOLVED_G4_MIXTURE",
                "serialized_stream": perturbation["stream"],
                "serialized_coordinate_step_count": serialized_steps,
                "measured_delta_D_component": delta_D,
                "measured_abs_D_per_serialized_coordinate_step": (
                    abs(delta_D) / serialized_steps
                ),
                "receiver_changed_camera_values": source["receiver_bijection"][
                    "changed_camera_values"
                ],
                "receiver_uint8_abs_step_sum": None,
                "scorer_sensitivity_D_per_uint8_step": None,
                "effective_quantum_D": None,
                "status": "BLOCKED_UINT8_ABS_STEP_HISTOGRAM_AND_G4_JOINT_ASSIGNMENT_ABSENT",
                "serialized_step_is_not_uint8_step": True,
                "actionable_for_tolerance_price": False,
                "score_claim": False,
            }
            component_evidence.append(component)
            for temporal_class in G4_TEMPORAL_CLASSES:
                bucket_rows.append(
                    {
                        **component,
                        "g4_temporal_class": temporal_class,
                        "measured_delta_D_component": None,
                        "measured_abs_D_per_serialized_coordinate_step": None,
                        "status": (
                            "BLOCKED_PER_DIMENSION_UINT8_EFFECTIVE_QUANTUM_AND_"
                            "CANDIDATE_DELTA_TO_G4_ASSIGNMENT_ABSENT"
                        ),
                    }
                )
    return {
        "schema": "ddm_rd1_effective_quantum_tolerance.v1",
        "law": (
            "q_eff[d] = one_realized_uint8_step[d] * "
            "abs(dD/d_uint8[d]); tolerance prices are expressed in q_eff[d], "
            "never in a uniform serialized-coordinate tolerance"
        ),
        "uniform_tolerance_allowed": False,
        "component_evidence": component_evidence,
        "bucket_rows": bucket_rows,
        "priced_bucket_count": 0,
        "blocker": (
            "DR2b retained changed-value counts but not the per-dimension "
            "receiver uint8 absolute-step histogram; G4 joint assignment is also absent"
        ),
        "operator_corollary": (
            "sub-quantum low-sensitivity dimensions may be free while a nominally "
            "equal Lane tolerance may span multiple effective quanta; this receipt "
            "does not classify either case without the missing histogram"
        ),
        "score_claim": False,
    }


def refine_dimensions(config_path: Path) -> dict[str, Any]:
    """Type the completed frontier duals without rerunning any λ point."""

    config = _read_json(config_path)
    if (
        config.get("schema") != "DDMRD1DimensionDualConfigV1"
        or config.get("execution_allowed") is not False
        or config.get("research_only") is not True
        or config.get("score_claim") is not False
        or config.get("pair_count") != 600
        or config.get("pointer_moved") is not False
    ):
        raise RD1MeasurementError("dimension-refinement authority contract differs")
    config_sha256 = _canonical_hash(config)
    source = _bound_json(
        str(config["source_pooled_receipt_path"]),
        str(config["source_pooled_receipt_sha256"]),
        "source pooled RD1 receipt",
    )
    if (
        source.get("schema") != "ddm_rd1_lambda_continuation_frontier_receipt.v1"
        or source.get("score_claim") is not False
        or source.get("pointer_moved") is not False
    ):
        raise RD1MeasurementError("source pooled RD1 receipt authority differs")
    continuation_sha256 = _canonical_hash(source["continuation"])
    if continuation_sha256 != config["source_continuation_sha256"]:
        raise RD1MeasurementError("source continuation changed; λ restart is forbidden")
    dr2b = _bound_json(
        str(config["dr2b_receipt_path"]),
        str(config["dr2b_receipt_sha256"]),
        "DR2b effective-quantum source",
    )
    g4 = _bound_json(
        str(config["g4_receipt_path"]),
        str(config["g4_receipt_sha256"]),
        "G4 temporal-class source",
    )
    observed_g4_classes = set(
        g4["summary"]["stationarity_decomposition"]["all"]["classes"]
    )
    if observed_g4_classes != set(G4_TEMPORAL_CLASSES):
        raise RD1MeasurementError("G4 temporal-class vocabulary differs")

    hull = tuple(_description_from_receipt(row) for row in source["supported_hull"])
    dimension_duals = typed_dimension_dual_report(hull)
    tolerance_report = _effective_quantum_tolerance_report(dr2b)
    output_root = _repo_path(str(config["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    supplement_path = output_root / "typed_dimension_duals_effective_quantum.json"
    supplement = {
        "schema": "ddm_rd1_dimension_duals_effective_quantum.v1",
        "typed_config_path": str(config_path.relative_to(REPO)),
        "typed_config_sha256": config_sha256,
        "source_pooled_receipt_path": config["source_pooled_receipt_path"],
        "source_pooled_receipt_sha256": config["source_pooled_receipt_sha256"],
        "source_continuation_sha256": continuation_sha256,
        "lambda_points_reused_without_restart": True,
        "dimension_duals": dimension_duals,
        "effective_quantum_tolerance": tolerance_report,
        "g4_source": {
            "path": config["g4_receipt_path"],
            "sha256": config["g4_receipt_sha256"],
            "classes": list(G4_TEMPORAL_CLASSES),
            "transfer_status": (
                "VOCABULARY_ONLY; G4 aggregate fractions are not assigned to RD1 "
                "candidate deltas"
            ),
        },
        "directive_consumed": {
            "utc": "2026-07-24T02:04:16Z",
            "application": (
                "pooled lambda retained as diagnostic only; complete "
                "stratum x scorer-visibility x G4 cube emitted; effective "
                "quantum pricing fails closed without per-dimension uint8 custody"
            ),
        },
        "evidence_axis": EVIDENCE_AXIS,
        "execution_allowed": False,
        "research_only": True,
        "score_claim": False,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "main_landing_review_required": True,
        "verdict": (
            "POSTSOLVE_DIMENSION_TYPING_COMPLETE; TRAIN_DECISION_PRICES_BLOCKED_"
            "PENDING_JOINT_G4_RATE_HOME_AND_UINT8_EFFECTIVE_QUANTA"
        ),
    }
    publish_immutable_json(supplement_path, supplement)
    supplement_sha256 = _sha256_file(supplement_path)

    refined = json.loads(json.dumps(source))
    aggregate_controls = refined.pop("duals")
    historical_solve_table = refined.pop("train_decision_SOLVE_table")
    refined["schema"] = "ddm_rd1_lambda_continuation_frontier_receipt.v2"
    refined["typed_dimension_config_path"] = str(config_path.relative_to(REPO))
    refined["typed_dimension_config_sha256"] = config_sha256
    refined["source_pooled_receipt_path"] = config["source_pooled_receipt_path"]
    refined["source_pooled_receipt_sha256"] = config["source_pooled_receipt_sha256"]
    refined["source_continuation_sha256"] = continuation_sha256
    refined["lambda_points_reused_without_restart"] = True
    refined["pooled_dual_status"] = (
        "VALID_SCALARIZATION_CONTROL_SUPERSEDED_FOR_TRAIN_DECISION_PRICING"
    )
    refined["aggregate_scalarization_controls"] = aggregate_controls
    refined["historical_pooled_train_decision_table"] = {
        "status": "SUPERSEDED_NONACTIONABLE",
        "rows": historical_solve_table,
    }
    refined["duals"] = dimension_duals["bucket_rows"]
    refined["dimension_dual_axes"] = dimension_duals["axes"]
    refined["dimension_dual_component_evidence"] = dimension_duals[
        "component_evidence"
    ]
    refined["dimension_dual_edge_summaries"] = dimension_duals["edge_summaries"]
    refined["effective_quantum_tolerance"] = tolerance_report
    refined["dimension_supplement"] = {
        "path": str(supplement_path.relative_to(REPO)),
        "sha256": supplement_sha256,
    }
    refined["train_decision_SOLVE_table"] = [
        {
            "bucket": (
                f"edge_{row['dual_index']}:"
                f"{row['left_candidate_id']}->{row['right_candidate_id']}"
            ),
            "SOLVE": row["SOLVE"],
            "lambda_bytes_per_D_dimension": None,
            "aggregate_lambda_control": row["aggregate_lambda_control"],
            "scope": (
                "train-decision price blocked; aggregate secant is diagnostic only"
            ),
        }
        for row in dimension_duals["edge_summaries"]
    ] + [
        {
            "bucket": "PER_DIMENSION_EFFECTIVE_QUANTUM",
            "SOLVE": "BLOCKED_UINT8_ABS_STEP_HISTOGRAM_AND_G4_JOINT_ASSIGNMENT",
            "lambda_bytes_per_D_dimension": None,
            "aggregate_lambda_control": None,
            "scope": tolerance_report["blocker"],
        }
    ]
    refined["directives_consumed"].append(supplement["directive_consumed"])
    refined["verdict"] = (
        f"{source['verdict']}; DIMENSION_TYPED_POSTSOLVE; "
        "TRAIN_DECISION_PRICES_BLOCKED_WITHOUT_JOINT_CUSTODY"
    )
    refined["verdict_scope"] = (
        f"{source['verdict_scope']}; pooled lambda remains a scalarization "
        "control only, not a per-dimension exchange rate"
    )
    receipt_path = output_root / "ddm_rd1_lambda_continuation_frontier_receipt_v3.json"
    publish_immutable_json(receipt_path, refined)
    receipt_sha256 = _sha256_file(receipt_path)

    typed_frontier_path = output_root / "typed_R_D_frontier_rows_v3.json"
    typed_frontier = {
        "schema": "ddm_rd1_typed_rate_distortion_rows.v2",
        "source_receipt_path": str(receipt_path.relative_to(REPO)),
        "source_receipt_sha256": receipt_sha256,
        "source_continuation_sha256": continuation_sha256,
        "lambda_points_reused_without_restart": True,
        "dimension_supplement": refined["dimension_supplement"],
        "objective": refined["objective"]["formula"],
        "rows": [
            {
                "lambda_index": row["lambda_index"],
                "lambda": row["lambda"],
                "candidate_id": row["selected_candidate_id"],
                "counted_bytes": row["counted_bytes"],
                "d_seg": row["d_seg"],
                "d_pose": row["d_pose"],
                "D_realized": row["D_realized"],
                "rate_term_25R": 25.0 * row["counted_bytes"] / 37_545_489,
                "S_composed": (
                    row["D_realized"]
                    + 25.0 * row["counted_bytes"] / 37_545_489
                ),
                "skeleton_bytes": row["skeleton_bytes"],
                "fiber_bytes": row["fiber_bytes"],
                "receiver_closure": row["receiver_closure"],
                "pair_count": row["pair_count"],
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            }
            for row in refined["continuation"]
        ],
        "interpretation": (
            "The λ continuation is byte-identical to the completed source solve. "
            "Per-dimension train pricing is carried by the supplement and fails "
            "closed where joint G4/rate-home or uint8-quantum custody is absent."
        ),
        "pointer": config["pointer"],
        "pointer_moved": False,
    }
    publish_immutable_json(typed_frontier_path, typed_frontier)

    metric_geometry = metric_active_continuation_geometry_report(
        refined["continuation"]
    )
    metric_supplement_path = (
        output_root / "metric_active_continuation_geometry.json"
    )
    metric_supplement = {
        **metric_geometry,
        "source_dimension_receipt_path": str(receipt_path.relative_to(REPO)),
        "source_dimension_receipt_sha256": receipt_sha256,
        "source_continuation_sha256": continuation_sha256,
        "directive_consumed": {
            "utc": "2026-07-24T02:27:12Z",
            "application": (
                "discrete neighbor traversal classified as graph topology with "
                "no state-space norm; continuous direction, trust, distance, "
                "and proposal ranking require measured Fisher/Pose/Bregman "
                "geometry; identity L2 is control-only"
            ),
        },
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    publish_immutable_json(metric_supplement_path, metric_supplement)
    metric_supplement_sha256 = _sha256_file(metric_supplement_path)

    metric_refined = json.loads(json.dumps(refined))
    metric_refined["schema"] = (
        "ddm_rd1_lambda_continuation_frontier_receipt.v3"
    )
    metric_refined["source_dimension_receipt_path"] = str(
        receipt_path.relative_to(REPO)
    )
    metric_refined["source_dimension_receipt_sha256"] = receipt_sha256
    metric_refined["metric_active_continuation_geometry"] = metric_geometry
    metric_refined["metric_geometry_supplement"] = {
        "path": str(metric_supplement_path.relative_to(REPO)),
        "sha256": metric_supplement_sha256,
    }
    metric_refined["directives_consumed"].append(
        metric_supplement["directive_consumed"]
    )
    metric_refined["verdict"] = (
        f"{refined['verdict']}; DISCRETE_GRAPH_GEOMETRY_VALID; "
        "CONTINUOUS_METRIC_ACTIVE_PROPOSALS_BLOCKED_PENDING_TENSOR_CUSTODY"
    )
    metric_refined["verdict_scope"] = (
        f"{refined['verdict_scope']}; no identity-L2 continuation verdict "
        "or continuous geometry is claimed"
    )
    metric_receipt_path = (
        output_root / "ddm_rd1_lambda_continuation_frontier_receipt_v4.json"
    )
    publish_immutable_json(metric_receipt_path, metric_refined)
    metric_receipt_sha256 = _sha256_file(metric_receipt_path)

    metric_typed_frontier = json.loads(json.dumps(typed_frontier))
    metric_typed_frontier["schema"] = (
        "ddm_rd1_typed_rate_distortion_rows.v3"
    )
    metric_typed_frontier["source_receipt_path"] = str(
        metric_receipt_path.relative_to(REPO)
    )
    metric_typed_frontier["source_receipt_sha256"] = metric_receipt_sha256
    metric_typed_frontier["metric_geometry_supplement"] = metric_refined[
        "metric_geometry_supplement"
    ]
    metric_typed_frontier["interpretation"] = (
        f"{typed_frontier['interpretation']} Discrete neighbor traversal is "
        "topological; continuous moves remain blocked until measured "
        "Fisher/Pose/Bregman and dual-metric readback custody exists."
    )
    metric_typed_frontier_path = (
        output_root / "typed_R_D_frontier_rows_v4.json"
    )
    publish_immutable_json(
        metric_typed_frontier_path,
        metric_typed_frontier,
    )

    second_order_geometry = second_order_metric_geometry_addendum_report(
        metric_geometry
    )
    second_order_supplement_path = (
        output_root / "metric_active_second_order_geometry.json"
    )
    second_order_supplement = {
        **second_order_geometry,
        "source_metric_receipt_path": str(
            metric_receipt_path.relative_to(REPO)
        ),
        "source_metric_receipt_sha256": metric_receipt_sha256,
        "source_continuation_sha256": continuation_sha256,
        "directive_consumed": {
            "utc": "2026-07-24T02:28:21Z",
            "application": (
                "quadratic regions require second-order geometry from step one; "
                "moves use rank-4 scorer class-pair coordinates; geometry "
                "ladders begin with the best measured form and descend only "
                "to labeled controls"
            ),
        },
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    publish_immutable_json(
        second_order_supplement_path,
        second_order_supplement,
    )
    second_order_supplement_sha256 = _sha256_file(
        second_order_supplement_path
    )

    second_order_refined = json.loads(json.dumps(metric_refined))
    second_order_refined["schema"] = (
        "ddm_rd1_lambda_continuation_frontier_receipt.v4"
    )
    second_order_refined["source_metric_receipt_path"] = str(
        metric_receipt_path.relative_to(REPO)
    )
    second_order_refined[
        "source_metric_receipt_sha256"
    ] = metric_receipt_sha256
    second_order_refined[
        "second_order_metric_geometry_addendum"
    ] = second_order_geometry
    second_order_refined["second_order_geometry_supplement"] = {
        "path": str(second_order_supplement_path.relative_to(REPO)),
        "sha256": second_order_supplement_sha256,
    }
    second_order_refined["directives_consumed"].append(
        second_order_supplement["directive_consumed"]
    )
    second_order_refined["verdict"] = (
        f"{metric_refined['verdict']}; SECOND_ORDER_SCORER_COORDINATE_"
        "OPTIMAL_FIRST_LADDER_REQUIRED"
    )
    second_order_refined["verdict_scope"] = (
        f"{metric_refined['verdict_scope']}; no first-order-naive, "
        "parameter-coordinate, or simple-first geometry ladder is admitted"
    )
    second_order_receipt_path = (
        output_root / "ddm_rd1_lambda_continuation_frontier_receipt_v5.json"
    )
    publish_immutable_json(
        second_order_receipt_path,
        second_order_refined,
    )
    second_order_receipt_sha256 = _sha256_file(
        second_order_receipt_path
    )

    second_order_typed_frontier = json.loads(
        json.dumps(metric_typed_frontier)
    )
    second_order_typed_frontier["schema"] = (
        "ddm_rd1_typed_rate_distortion_rows.v4"
    )
    second_order_typed_frontier["source_receipt_path"] = str(
        second_order_receipt_path.relative_to(REPO)
    )
    second_order_typed_frontier[
        "source_receipt_sha256"
    ] = second_order_receipt_sha256
    second_order_typed_frontier[
        "second_order_geometry_supplement"
    ] = second_order_refined["second_order_geometry_supplement"]
    second_order_typed_frontier["interpretation"] = (
        f"{metric_typed_frontier['interpretation']} Any future continuous "
        "ladder starts second-order in scorer coordinates when measured; "
        "simpler or identity geometry is control-only."
    )
    second_order_typed_frontier_path = (
        output_root / "typed_R_D_frontier_rows_v5.json"
    )
    publish_immutable_json(
        second_order_typed_frontier_path,
        second_order_typed_frontier,
    )
    return {
        "receipt_path": str(second_order_receipt_path),
        "receipt_sha256": second_order_receipt_sha256,
        "typed_frontier_path": str(second_order_typed_frontier_path),
        "typed_frontier_sha256": _sha256_file(
            second_order_typed_frontier_path
        ),
        "source_metric_receipt_path": str(metric_receipt_path),
        "source_metric_receipt_sha256": metric_receipt_sha256,
        "source_dimension_receipt_path": str(receipt_path),
        "source_dimension_receipt_sha256": receipt_sha256,
        "dimension_supplement_path": str(supplement_path),
        "dimension_supplement_sha256": supplement_sha256,
        "metric_geometry_supplement_path": str(metric_supplement_path),
        "metric_geometry_supplement_sha256": metric_supplement_sha256,
        "second_order_geometry_supplement_path": str(
            second_order_supplement_path
        ),
        "second_order_geometry_supplement_sha256": (
            second_order_supplement_sha256
        ),
        "source_continuation_sha256": continuation_sha256,
        "lambda_points_reused_without_restart": True,
        "typed_dual_bucket_count": len(dimension_duals["bucket_rows"]),
        "actionable_dual_bucket_count": dimension_duals["actionable_bucket_count"],
        "priced_tolerance_bucket_count": tolerance_report["priced_bucket_count"],
        "continuous_metric_geometry_actionable": metric_geometry[
            "continuous_proposal_geometry_contract"
        ]["actionable"],
        "second_order_geometry_actionable": second_order_geometry[
            "future_continuous_move_contract"
        ]["actionable"],
        "verdict": second_order_refined["verdict"],
    }


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if (
        config.get("schema") != "DDMRD1LambdaContinuationConfigV1"
        or config.get("execution_allowed") is not True
        or config.get("research_only") is not True
        or config.get("score_claim") is not False
        or config.get("pair_count") != 600
        or config.get("scorer_threads") != 4
    ):
        raise RD1MeasurementError("typed execution/authority contract differs")
    config_sha256 = _canonical_hash(config)
    checkpoint_root = Path(str(config["checkpoint_root"]))
    output_root = _repo_path(str(config["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    storage = _storage_preflight(checkpoint_root)

    v19_receipt = _bound_json(
        str(config["v19c_receipt_path"]),
        str(config["v19c_receipt_sha256"]),
        "V19C receipt",
    )
    menu_receipt = _bound_json(
        str(config["menu1_receipt_path"]),
        str(config["menu1_receipt_sha256"]),
        "Menu1 receipt",
    )
    dr2b_receipt = _bound_json(
        str(config["dr2b_receipt_path"]),
        str(config["dr2b_receipt_sha256"]),
        "DR2b receipt",
    )
    c1_prepare = _bound_json(
        str(config["c1_prepare_receipt_path"]),
        str(config["c1_prepare_receipt_sha256"]),
        "C1 prepare receipt",
    )
    c1_inflate = _bound_json(
        str(config["c1_inflate_receipt_path"]),
        str(config["c1_inflate_receipt_sha256"]),
        "C1 inflate receipt",
    )
    c1_preserved = _bound_json(
        str(config["c1_preserved_eval_path"]),
        str(config["c1_preserved_eval_sha256"]),
        "C1 preserved evaluation receipt",
    )

    large_input_hashes = [
        _file_hash_checkpoint(
            Path(str(config["c1_archive_path"])),
            expected_sha256=str(config["c1_archive_sha256"]),
            checkpoint_root=checkpoint_root,
            label="C1 exact solved archive",
        ),
        _file_hash_checkpoint(
            Path(str(config["c1_raw_path"])),
            expected_sha256=str(config["c1_raw_sha256"]),
            checkpoint_root=checkpoint_root,
            label="C1 receiver output raw",
        ),
        _file_hash_checkpoint(
            Path(str(config["target_cache_path"])),
            expected_sha256=str(config["target_cache_sha256"]),
            checkpoint_root=checkpoint_root,
            label="n600 frozen scorer target cache",
        ),
    ]
    if Path(str(config["target_cache_path"])).stat().st_size != int(config["target_cache_bytes"]):
        raise RD1MeasurementError("target cache byte count differs")

    c1_local = _verify_c1_scorers(
        config,
        config_sha256=config_sha256,
        checkpoint_root=checkpoint_root,
    )
    preserved_c1_d_seg = float(c1_preserved["avg_segnet_dist"])
    preserved_c1_d_pose = float(c1_preserved["avg_posenet_dist"])
    c1_axis_crosscheck = {
        "preserved_axis": "[contest-CPU Linux x86_64]",
        "preserved_d_seg_display": preserved_c1_d_seg,
        "preserved_d_pose_display": preserved_c1_d_pose,
        "fresh_axis": EVIDENCE_AXIS,
        "fresh_d_seg": float(c1_local["d_seg"]),
        "fresh_d_pose": float(c1_local["d_pose"]),
        "fresh_minus_preserved_display_d_seg": (float(c1_local["d_seg"]) - preserved_c1_d_seg),
        "fresh_minus_preserved_display_d_pose": (float(c1_local["d_pose"]) - preserved_c1_d_pose),
        "status": (
            "DISPLAY_ROUNDING_MATCH"
            if round(float(c1_local["d_seg"]), 8) == preserved_c1_d_seg
            and round(float(c1_local["d_pose"]), 8) == preserved_c1_d_pose
            else "MEASURED_AXIS_OR_BATCH_GEOMETRY_DRIFT"
        ),
        "policy": (
            "use the fresh local four-thread values on the RD1 advisory curve; "
            "retain the contest row as a separate cross-axis display-only check"
        ),
    }
    if (
        c1_prepare["archive_sha256"] != config["c1_archive_sha256"]
        or c1_inflate["archive_sha256"] != config["c1_archive_sha256"]
        or c1_inflate["raw_sha256"] != config["c1_raw_sha256"]
        or c1_inflate["both_planes_exact"] is not True
        or c1_inflate["strict_packet_reencode_identical"] is not True
    ):
        raise RD1MeasurementError("C1 receiver custody differs")

    v19_candidates = _load_v19_candidates(config, v19_receipt)
    current_id = f"v19c_admit_{int(v19_receipt['curve']['n600_per_admitted_move'][-1]['admission_index']):04d}"
    by_id = {candidate.candidate_id: candidate for candidate in v19_candidates}
    if current_id not in by_id:
        raise RD1MeasurementError("V19C current describe-line candidate is absent")
    current = by_id[current_id]
    if (
        current.source_sha256 != config["v19c_current_archive_sha256"]
        or current.counted_bytes != v19_receipt["curve"]["n600_endpoint"]["archive_bytes"]
    ):
        raise RD1MeasurementError("V19C current archive custody differs")
    menu_candidates, base_aggregate = _load_menu_candidates(config, menu_receipt, v19_current=current)
    current = _replace_per_class(current, base_aggregate["per_class"])
    v19_candidates = [current if candidate.candidate_id == current_id else candidate for candidate in v19_candidates]

    c1_archive = Path(str(config["c1_archive_path"]))
    c1_streams = _split_c1_archive(
        c1_archive,
        packet_sha256=str(c1_prepare["packet_sha256"]),
        packet_bytes=int(c1_prepare["packet_bytes"]),
    )
    c1_candidate = MeasuredDescription(
        candidate_id="c1_exact_solved_n600",
        counted_bytes=int(config["c1_archive_bytes"]),
        d_seg=float(c1_local["d_seg"]),
        d_pose=float(c1_local["d_pose"]),
        coded_streams=c1_streams,
        source_artifact=str(c1_archive),
        source_sha256=str(config["c1_archive_sha256"]),
        receiver_closure="archive_receiver_closed",
        per_class=c1_local["per_class"],
        metadata={
            "source_family": "C1 exact two-plane",
            "mechanism_bucket": "EXACT_FULL_RANK_UINT8_LATTICE",
            "fresh_local_scorer_replay": True,
            "both_planes_exact": True,
        },
    )
    candidates = [*v19_candidates, *menu_candidates, c1_candidate]
    nondominated = pareto_nondominated(candidates)
    hull = lower_supported_hull(candidates)
    lambdas = geometric_curvature_ladder(hull, minimum_points=10, maximum_points=12)
    continuation = continuation_rows(
        candidates,
        lambdas,
        seed_candidate_id=current_id,
    )
    lambda_checkpoint_root = checkpoint_root / "02_lambda_continuation"
    for row in continuation:
        publish_immutable_json(
            lambda_checkpoint_root / f"lambda_{int(row['lambda_index']):02d}.json",
            {
                "schema": "ddm_rd1_lambda_checkpoint.v1",
                "typed_config_sha256": config_sha256,
                **row,
            },
        )
    duals = discrete_dual_rows(hull)
    knee = normalized_knee(hull)
    knee_artifact = _build_knee_bundle(
        knee,
        output_root=output_root,
        config_sha256=config_sha256,
    )
    dr2b = _dr2b_crosscheck(dr2b_receipt)
    solve_table = _train_decision_solve_table(duals, dr2b=dr2b)
    hull_ids = {row.candidate_id for row in hull}
    nondominated_ids = {row.candidate_id for row in nondominated}
    current_scalarizable = current_id in hull_ids
    r6_candidate = knee.counted_bytes <= 200_000 and knee.d_seg <= 0.00116 and knee.d_pose <= 0.00161
    receipt = {
        "schema": "ddm_rd1_lambda_continuation_frontier_receipt.v1",
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "typed_config_path": str(config_path.relative_to(REPO)),
        "typed_config_sha256": config_sha256,
        "git_sha_before_landing": _git_sha(),
        "objective": {
            "formula": "counted_bytes(x) + lambda * (100*d_seg(x) + sqrt(10*d_pose(x)))",
            "optimization_domain": config["optimization_domain"],
            "global_uint8_lattice_optimality_claim": False,
            "restricted_domain_candidate_count": len(candidates),
            "realized_only": True,
            "real_coder_only": True,
            "own_stored_problem_only": True,
            "donor_conditioned": False,
        },
        "anchors": {
            "describe_line_control": {
                **current.to_dict(),
                "requested_role": "lambda_0_control",
                "scalarizable_supported_point": current_scalarizable,
                "status": (
                    "SUPPORTED" if current_scalarizable else "UNSUPPORTED_PARETO_CONTROL_CORRECTED_BY_NEIGHBOR_PATH"
                ),
            },
            "lambda_infinity_exact": {
                **c1_candidate.to_dict(),
                "requested_role": "lambda_infinity_limit",
                "fresh_local_n600_scorer_recomputed": True,
                "archive_and_receiver_reverified": True,
                "preserved_exact_row_cross_axis_check": c1_axis_crosscheck,
            },
        },
        "input_hashes": large_input_hashes,
        "storage_preflight": storage,
        "candidate_domain": [candidate.to_dict() for candidate in candidates],
        "pareto_nondominated_candidate_ids": [candidate.candidate_id for candidate in nondominated],
        "supported_hull": [candidate.to_dict() for candidate in hull],
        "unsupported_nondominated_candidate_ids": sorted(nondominated_ids - hull_ids),
        "lambda_ladder": list(lambdas),
        "continuation": list(continuation),
        "per_lambda_checkpoint_root": str(lambda_checkpoint_root),
        "duals": list(duals),
        "dr2b_crosscheck": dr2b,
        "train_decision_SOLVE_table": solve_table,
        "knee": {
            **knee.to_dict(),
            "selection_law": "maximum normalized elbow in log1p(rate) versus D",
            "full_description_artifact": knee_artifact,
            "R6_CANDIDATE": r6_candidate,
            "R6_flag_policy": "flag only; no dispatch",
            "box": {
                "archive_bytes_max": 200_000,
                "d_seg_max": 0.00116,
                "d_pose_max": 0.00161,
            },
        },
        "ms1_reconciliation": {
            "status": "NOT_CONSUMED_SIBLING_STILL_ACTIVE_AT_MEASUREMENT_BUILD",
            "pool": "solve_frontier",
            "competes_with_pool": "solver_member_selection",
            "combination": "competitive_never_additive",
            "rule": (
                "A future donor-free MS1 archive-closed point may enter this finite "
                "domain and be reranked; no conditional-coder diagnostic is imported."
            ),
        },
        "negatives": [
            {
                "scope": "FORMULATION",
                "region": "weighted-sum scalarization at the V19C current control",
                "finding": (
                    "The current V19C point is Pareto-nondominated but unsupported "
                    "after the measured Menu1 exception pool is admitted, so no "
                    "lambda makes it x*(lambda) in this finite domain."
                ),
                "families_remaining_open": True,
            },
            {
                "scope": "FORMULATION",
                "region": "Menu1 deployment closure",
                "finding": (
                    "Menu1 points are exact n600 measurement-harness receiver rows, "
                    "not contest archive-receiver-closed candidates."
                ),
                "families_remaining_open": True,
            },
            {
                "scope": "FORMULATION",
                "region": "global uint8 lattice",
                "finding": (
                    "Full-rank verification covers the SHA-custodied measured domain "
                    "only; no global-lattice optimum is claimed."
                ),
                "families_remaining_open": True,
            },
        ],
        "triality": {
            "DAG": ".omx/research/ddm_rd1_lambda_continuation_frontier_DAG_FEED_20260724.md",
            "DSL": str(config_path.relative_to(REPO)),
            "equation_id": "ddm_restricted_realized_lambda_continuation_v1",
            "equation_module": ("tac.canonical_equations.ddm_lambda_continuation_frontier_20260724"),
        },
        "directives_consumed": [
            {
                "utc": "2026-07-19T19:42:07Z",
                "application": (
                    "full measured marginal rank; same-pool candidates compete; "
                    "DR2b free-tolerance rows cross-checked without transfer"
                ),
            },
            {
                "utc": "2026-07-19T19:48:01Z",
                "application": (
                    "realized frozen-scorer metric retained; no Euclidean/Fourier "
                    "proxy or unmeasured inner-Jacobian claim"
                ),
            },
            {
                "utc": "2026-07-24T01:12:39Z",
                "application": (
                    "delegated RD1 authority: no paid dispatch, exact evaluator, "
                    "training, pointer mutation, or donor conditioning"
                ),
            },
        ],
        "evidence_axis": EVIDENCE_AXIS,
        "execution_allowed": True,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "main_landing_review_required": True,
        "verdict": (
            "MEASURED_RESTRICTED_N600_LAMBDA_FRONTIER; "
            "V19C_CURRENT_UNSUPPORTED; KNEE_OUTSIDE_R6_BOX; "
            "GLOBAL_LATTICE_AND_MENU_ARCHIVE_CLOSURE_OPEN"
        ),
        "verdict_scope": (
            "FORMULATION: SHA-custodied V19C/Menu1/C1 measured n600 description "
            "domain on macOS CPU frozen scorers; families and paradigm remain open"
        ),
    }
    receipt_path = output_root / "ddm_rd1_lambda_continuation_frontier_receipt_v2.json"
    publish_immutable_json(receipt_path, receipt)
    receipt_sha256 = _sha256_file(receipt_path)
    typed_frontier_path = output_root / "typed_R_D_frontier_rows.json"
    typed_frontier = {
        "schema": "ddm_rd1_typed_rate_distortion_rows.v1",
        "source_receipt_path": str(receipt_path.relative_to(REPO)),
        "source_receipt_sha256": receipt_sha256,
        "objective": receipt["objective"]["formula"],
        "rows": [
            {
                "lambda_index": row["lambda_index"],
                "lambda": row["lambda"],
                "candidate_id": row["selected_candidate_id"],
                "counted_bytes": row["counted_bytes"],
                "d_seg": row["d_seg"],
                "d_pose": row["d_pose"],
                "D_realized": row["D_realized"],
                "rate_term_25R": 25.0 * row["counted_bytes"] / 37_545_489,
                "S_composed": (row["D_realized"] + 25.0 * row["counted_bytes"] / 37_545_489),
                "skeleton_bytes": row["skeleton_bytes"],
                "fiber_bytes": row["fiber_bytes"],
                "receiver_closure": row["receiver_closure"],
                "pair_count": row["pair_count"],
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            }
            for row in continuation
        ],
        "interpretation": (
            "S_composed is arithmetic only on local advisory components; it is not "
            "a contest score claim. Repeated candidates at different lambda values "
            "are retained as continuation checkpoints rather than fabricated points."
        ),
        "pointer": config["pointer"],
        "pointer_moved": False,
    }
    publish_immutable_json(typed_frontier_path, typed_frontier)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "typed_frontier_path": str(typed_frontier_path),
        "typed_frontier_sha256": _sha256_file(typed_frontier_path),
        "candidate_count": len(candidates),
        "hull_count": len(hull),
        "lambda_count": len(continuation),
        "knee_candidate_id": knee.candidate_id,
        "R6_CANDIDATE": r6_candidate,
        "verdict": receipt["verdict"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dimension-config", type=Path, default=DEFAULT_DIMENSION_CONFIG)
    parser.add_argument("--refine-dimensions-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        refine_dimensions(args.dimension_config.resolve())
        if args.refine_dimensions_only
        else run(args.config.resolve())
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
