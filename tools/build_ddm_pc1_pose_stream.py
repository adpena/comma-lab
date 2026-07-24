#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and n600-measure the typed counted DDM PC1 pose-stream member."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.canonical_equations.ddm_pc1_pose_stream_20260724 import (  # noqa: E402
    admission_fence,
)
from tac.optimization.ddm_pc1_pose_stream import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    PAIR_H,
    PAIR_W,
    DDMPC1TrainableParameterMapV1,
    PC1PosePacketV1,
    active_tube_quadratic,
    build_counted_composition_archive,
    conditional_score_delta,
    fresh_pose_initialization,
    make_inactive_packet,
    output_effect_owners,
    parse_counted_composition_archive,
    parse_pc1_packet,
    receive_pc1_camera_pairs,
    serialize_pc1_packet,
    sha256_bytes,
    solved_plane_yuv6_target,
    verify_unique_output_effect_owners,
)
from tac.optimization.ddm_runtime_sensitivity import (  # noqa: E402
    composite_r_support_mask,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    receive_ws1_warm_start_archive,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
)
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    _config_and_inputs,
    _forward,
    _load_models,
)

CONFIG_SCHEMA = "DDMPC1PoseStreamAdmissionConfigV1"
RECEIPT_SCHEMA = "ddm_pc1_pose_stream_admission.v1"
ROW_SCHEMA = "ddm_pc1_pose_stream_batch32_row.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
CAMERA_PAIR_SHAPE = (2, CAMERA_H, CAMERA_W, 3)
PAIR_COUNT = 600
SCORER_BATCH = 32
SCORER_THREADS = 4
VERDICT_SCOPE = (
    "INSTANCE: fresh counted PC1 pose stream composed independently with exact W_seg "
    "and W_joint parents, n600 batch32 macOS-CPU frozen-scorer advisory only; no "
    "contest eval, promotion, dispatch, frontier mutation, or active-tube claim"
)


class PC1BuildError(RuntimeError):
    """Custody, storage, receiver, or scorer admission failed closed."""


class ParentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    raw_stage_root: str
    expected_d_seg: float = Field(ge=0.0)
    expected_d_pose: float = Field(ge=0.0)


class PC1BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMPC1PoseStreamAdmissionConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str
    lane_id: Literal["lane_ddm_pc1_pose_stream_admission_20260724"]
    delegation_checkpoint_key: Literal["codex_delegate:ddm_pc1_pose_stream_admission:20260724T191805Z"]
    authority_path: str
    authority_sha256: Literal["f54aab6f763ea6a0c5cd09a837f379495f3871c15207e054d123af0a0a851842"]
    authority_bytes: Literal[7111]
    own_python: str
    output_root: str
    ms4d_pose_metric_path: str
    ms4d_pose_metric_sha256: Literal["5e06cc78711a6ca6984c907600a25816cdecc6239903f782d85bcf9473a8f1bc"]
    menu1_config_path: str
    menu1_config_sha256: Literal["b9fed2b1537b92a2b02d0525cb4d9175d0704e7c4d0c0efd383c6dd818fdb2c7"]
    ws2_receipt_path: str
    ws2_receipt_sha256: Literal["05581b02cc6ce789b6219302ebd888f1665ab4c3882038ce29e9be18f6174ea1"]
    parents: dict[Literal["W_seg", "W_joint"], ParentConfig]
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[32] = 32
    scorer_threads: Literal[4] = 4
    seed: Literal[0] = 0
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _bound_bytes(path_value: str, sha256: str, label: str) -> bytes:
    path = _resolve(path_value)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise PC1BuildError(f"{label} SHA-256 differs")
    return payload


def _bound_json(path_value: str, sha256: str, label: str) -> dict[str, Any]:
    payload = _bound_bytes(path_value, sha256, label)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PC1BuildError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise PC1BuildError(f"{label} must be one JSON object")
    return value


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PC1BuildError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_json(path: Path, value: Any) -> None:
    _publish_bytes(path, _canonical_json(value))


def _load_config(path: Path) -> tuple[PC1BuildConfig, str]:
    payload = path.read_bytes()
    try:
        config = PC1BuildConfig.model_validate_json(payload, strict=True)
    except Exception as exc:
        raise PC1BuildError("typed PC1 config validation failed") from exc
    canonical = _canonical_json(config.model_dump(mode="json", by_alias=True))
    if payload != canonical:
        raise PC1BuildError("typed PC1 config is not canonical JSON")
    expected_python = Path(config.own_python).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise PC1BuildError(f"PC1 must run in its owned SSD venv: {expected_python}; got {sys.executable}")
    return config, hashlib.sha256(payload).hexdigest()


def _storage_preflight(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    required_free_bytes = 20 * 1024**3
    if usage.free < required_free_bytes:
        raise PC1BuildError("SSD storage preflight failed closed")
    if not str(output_root.resolve()).startswith("/Volumes/VertigoDataTier/pact/"):
        raise PC1BuildError("PC1 output root must stay on the primary SSD tier")
    return {
        "cleanup": (
            "candidate archives and immutable JSON batch checkpoints retained; "
            "parent raw stages remain external certified inputs; no scratch survives success"
        ),
        "free_bytes_gte_required": True,
        "output_root": str(output_root),
        "required_free_bytes": required_free_bytes,
        "status": "PASS",
        "tier": "/Volumes/VertigoDataTier/pact",
    }


def _load_metric(config: PC1BuildConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metric = _bound_json(
        config.ms4d_pose_metric_path,
        config.ms4d_pose_metric_sha256,
        "MS4d pose metric",
    )
    rows = metric.get("rows")
    if (
        metric.get("schema") != "ddm_pose_metric_custody.v1"
        or metric.get("pair_count") != PAIR_COUNT
        or metric.get("scorer_batch_size") != SCORER_BATCH
        or not isinstance(rows, list)
        or len(rows) != PAIR_COUNT
    ):
        raise PC1BuildError("MS4d pose metric custody/schema differs")
    if [row.get("pair_id") for row in rows] != list(range(PAIR_COUNT)):
        raise PC1BuildError("MS4d pose metric pair ordering differs")
    centers = np.asarray([row["center"] for row in rows], dtype=np.float64)
    factors = np.asarray([row["low_rank_factors"] for row in rows], dtype=np.float64)
    if centers.shape != (PAIR_COUNT, 6) or factors.shape != (PAIR_COUNT, 6, 6):
        raise PC1BuildError("MS4d pose metric tensor geometry differs")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(factors)):
        raise PC1BuildError("MS4d pose metric contains nonfinite values")
    return centers, factors, metric


def _validate_parent_custody(
    config: PC1BuildConfig,
    ws2_receipt: dict[str, Any],
) -> None:
    if set(config.parents) != {"W_seg", "W_joint"}:
        raise PC1BuildError("PC1 requires exactly W_seg and W_joint parents")
    endpoints = ws2_receipt.get("fresh_batch32_endpoints", {})
    archives = ws2_receipt.get("archive_custody", {})
    for name, parent in config.parents.items():
        endpoint = endpoints.get(name, {})
        archive = archives.get(name, {})
        if (
            endpoint.get("archive_sha256") != parent.archive_sha256
            or endpoint.get("archive_bytes") != parent.archive_bytes
            or float(endpoint.get("d_seg", -1.0)) != parent.expected_d_seg
            or float(endpoint.get("d_pose", -1.0)) != parent.expected_d_pose
            or archive.get("archive_sha256") != parent.archive_sha256
        ):
            raise PC1BuildError(f"{name} settled WS2 custody differs")


def _load_parent_stage(
    *,
    parent: ParentConfig,
    start: int,
    stop: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    root = Path(parent.raw_stage_root)
    chunks: list[np.ndarray] = []
    custody_rows: list[dict[str, Any]] = []
    chunk_start = start
    while chunk_start < stop:
        natural_start = (chunk_start // 16) * 16
        natural_stop = min(natural_start + 16, PAIR_COUNT)
        if natural_start != chunk_start:
            raise PC1BuildError("parent stage range must start on a preserved 16-pair boundary")
        take_stop = min(natural_stop, stop)
        stem = f"base_pairs_{natural_start:04d}_{natural_stop:04d}"
        raw_path = root / f"{stem}.raw"
        sidecar_path = root / f"{stem}.json"
        try:
            sidecar = json.loads(sidecar_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise PC1BuildError(f"parent stage sidecar unavailable: {sidecar_path}") from exc
        expected_bytes = (natural_stop - natural_start) * int(np.prod(CAMERA_PAIR_SHAPE))
        if (
            sidecar
            != {
                "bytes": expected_bytes,
                "manifest_sha256": parent.archive_sha256,
                "pair_start": natural_start,
                "pair_stop": natural_stop,
                "sha256": sidecar.get("sha256"),
            }
            or not isinstance(sidecar.get("sha256"), str)
            or len(sidecar["sha256"]) != 64
        ):
            raise PC1BuildError(f"parent stage sidecar schema differs: {sidecar_path}")
        raw_bytes, raw_sha256 = _sha256_file(raw_path)
        if raw_bytes != expected_bytes or raw_sha256 != sidecar["sha256"]:
            raise PC1BuildError(f"parent stage raw custody differs: {raw_path}")
        memmap = np.memmap(
            raw_path,
            mode="r",
            dtype=np.uint8,
            shape=(natural_stop - natural_start, *CAMERA_PAIR_SHAPE),
        )
        chunks.append(np.array(memmap[: take_stop - natural_start], copy=True, order="C"))
        custody_rows.append(
            {
                "bytes": raw_bytes,
                "pair_range": [natural_start, natural_stop],
                "path": str(raw_path),
                "sha256": raw_sha256,
            }
        )
        chunk_start = take_stop
    camera = np.concatenate(chunks, axis=0)
    if camera.shape != (stop - start, *CAMERA_PAIR_SHAPE):
        raise PC1BuildError("assembled parent stage geometry differs")
    return np.ascontiguousarray(camera), custody_rows


def _checkpoint_is_reusable(
    path: Path,
    *,
    parent_name: str,
    packet_sha256: str,
    start: int,
    stop: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise PC1BuildError(f"preserved PC1 checkpoint is malformed: {path}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != ROW_SCHEMA
        or value.get("parent") != parent_name
        or value.get("packet_sha256") != packet_sha256
        or value.get("pair_range") != [start, stop]
        or value.get("evidence_axis") != EVIDENCE_AXIS
        or value.get("score_claim") is not False
    ):
        raise PC1BuildError(f"preserved PC1 checkpoint custody differs: {path}")
    return value


def _environment_receipt(config: PC1BuildConfig) -> dict[str, Any]:
    runtime_distributions = (
        "Brotli",
        "einops",
        "numpy",
        "opencv-python-headless",
        "pydantic",
        "safetensors",
        "scipy",
        "segmentation-models-pytorch",
        "timm",
        "torch",
        "torchvision",
    )
    return {
        "configured_python": config.own_python,
        "executable": str(Path(sys.executable).resolve()),
        "runtime_distributions": {name: importlib.metadata.version(name) for name in runtime_distributions},
        "seed": config.seed,
        "sys_prefix": sys.prefix,
    }


def _measure_parent(
    *,
    name: str,
    parent: ParentConfig,
    packet: PC1PosePacketV1,
    nonzero_probe_packet: PC1PosePacketV1,
    output_root: Path,
    segnet: Any,
    posenet: Any,
    labels: np.ndarray,
    target_poses: np.ndarray,
    metric_centers: np.ndarray,
    metric_factors: np.ndarray,
    tube_radius: float,
) -> dict[str, Any]:
    archive_bytes = Path(parent.archive_path).read_bytes()
    if len(archive_bytes) != parent.archive_bytes or hashlib.sha256(archive_bytes).hexdigest() != parent.archive_sha256:
        raise PC1BuildError(f"{name} exact archive custody differs")
    receiver = receive_ws1_warm_start_archive(archive_bytes)
    if receiver.archive != archive_bytes:
        raise PC1BuildError(f"{name} receiver did not preserve exact parent bytes")
    try:
        movable_layer = next(layer for layer in receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise PC1BuildError(f"{name} receiver has no Movable layer") from exc

    complete_archive = build_counted_composition_archive(
        parent_archive=archive_bytes,
        parent_sha256=parent.archive_sha256,
        packet=packet,
    )
    parsed_parent, parsed_packet, manifest = parse_counted_composition_archive(complete_archive)
    if parsed_parent != archive_bytes or serialize_pc1_packet(parsed_packet) != serialize_pc1_packet(packet):
        raise PC1BuildError(f"{name} complete archive exact parse-back differs")
    archive_path = output_root / "archives" / f"{name}_plus_PC1.zip"
    _publish_bytes(archive_path, complete_archive)

    checkpoint_root = output_root / "checkpoints" / name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    packet_digest = sha256_bytes(serialize_pc1_packet(packet))
    rows: list[dict[str, Any]] = []
    first_batch_replay: dict[str, Any] | None = None
    for start in range(0, PAIR_COUNT, SCORER_BATCH):
        stop = min(start + SCORER_BATCH, PAIR_COUNT)
        row_path = checkpoint_root / f"batch_{start:04d}_{stop:04d}.json"
        preserved = _checkpoint_is_reusable(
            row_path,
            parent_name=name,
            packet_sha256=packet_digest,
            start=start,
            stop=stop,
        )
        if preserved is not None:
            rows.append(preserved)
            print(f"[PC1] {name} scorer {start:04d}:{stop:04d} preserved", flush=True)
            continue

        parent_camera, parent_stage_custody = _load_parent_stage(
            parent=parent,
            start=start,
            stop=stop,
        )
        pair_ids = list(range(start, stop))
        movable_masks = np.stack(
            [
                receiver._mask_for_layer(
                    movable_layer,
                    pair_id,
                    replace_g1_movable=True,
                )
                for pair_id in pair_ids
            ],
            axis=0,
        ).astype(np.bool_)
        candidate_camera = receive_pc1_camera_pairs(
            parent_camera=parent_camera,
            packet=packet,
            pair_ids=pair_ids,
            movable_masks=movable_masks,
        )
        inactive_camera = receive_pc1_camera_pairs(
            parent_camera=parent_camera,
            packet=make_inactive_packet(packet),
            pair_ids=pair_ids,
            movable_masks=movable_masks,
        )
        inactive_identity = np.array_equal(parent_camera, inactive_camera)
        if not inactive_identity:
            raise PC1BuildError(f"{name} inactive output identity failed")
        solved_plane_target = solved_plane_yuv6_target(parent_camera)
        cells, pose6 = _forward(segnet, posenet, candidate_camera)
        target = labels[start:stop]
        target_pose = target_poses[start:stop]
        per_class = {
            class_name: {
                "errors": int(np.count_nonzero((cells != target) & (target == class_id))),
                "sites": int(np.count_nonzero(target == class_id)),
            }
            for class_id, class_name in enumerate(CLASS_ORDER)
        }
        tube_cost = active_tube_quadratic(
            candidate_pose6=pose6,
            centers=metric_centers[start:stop],
            low_rank_factors=metric_factors[start:stop],
        )
        output_support = composite_r_support_mask(
            segnet=segnet,
            baseline_camera=parent_camera,
            perturbed_camera=candidate_camera,
        )
        frame_changed_values = [
            int(np.count_nonzero(candidate_camera[:, frame] != parent_camera[:, frame])) for frame in range(2)
        ]
        if any(value <= 0 for value in frame_changed_values) or int(np.count_nonzero(output_support)) <= 0:
            raise PC1BuildError(f"{name} active receiver has no composite-R support")
        causal_support_cells: int | None = None
        if start == 0:
            nonzero_probe_camera = receive_pc1_camera_pairs(
                parent_camera=parent_camera,
                packet=nonzero_probe_packet,
                pair_ids=pair_ids,
                movable_masks=movable_masks,
            )
            causal_support = composite_r_support_mask(
                segnet=segnet,
                baseline_camera=candidate_camera,
                perturbed_camera=nonzero_probe_camera,
            )
            causal_support_cells = int(np.count_nonzero(causal_support))
            if causal_support_cells <= 0:
                raise PC1BuildError(f"{name} nonzero q failed the landed MS6 causal composite-R probe")
            replay = receive_pc1_camera_pairs(
                parent_camera=parent_camera,
                packet=parse_pc1_packet(serialize_pc1_packet(packet)),
                pair_ids=pair_ids,
                movable_masks=movable_masks,
            )
            first_batch_replay = {
                "camera_sha256": sha256_bytes(candidate_camera.tobytes(order="C")),
                "exact_equal": bool(np.array_equal(candidate_camera, replay)),
                "pair_range": [start, stop],
                "replay_camera_sha256": sha256_bytes(replay.tobytes(order="C")),
            }
            if not first_batch_replay["exact_equal"]:
                raise PC1BuildError(f"{name} deterministic first-batch replay differs")
        row = {
            "active_tube_cost_max": float(np.max(tube_cost)),
            "active_tube_cost_sum": float(np.sum(tube_cost, dtype=np.float64)),
            "active_tube_membership_count_observed": int(np.count_nonzero(tube_cost <= tube_radius * tube_radius)),
            "candidate_camera_sha256": sha256_bytes(candidate_camera.tobytes(order="C")),
            "errors": int(np.count_nonzero(cells != target)),
            "evidence_axis": EVIDENCE_AXIS,
            "frame_changed_values": frame_changed_values,
            "inactive_output_byte_identity": inactive_identity,
            "ms6_nonzero_q_causal_support_cells": causal_support_cells,
            "output_composite_r_support_cells": int(np.count_nonzero(output_support)),
            "packet_sha256": packet_digest,
            "pair_range": [start, stop],
            "parent": name,
            "parent_camera_sha256": sha256_bytes(parent_camera.tobytes(order="C")),
            "parent_stage_custody": parent_stage_custody,
            "per_class": per_class,
            "pose_coordinates": int(pose6.size),
            "pose_squared_error_sum": float(np.square(pose6 - target_pose).sum(dtype=np.float64)),
            "schema": ROW_SCHEMA,
            "score_claim": False,
            "solved_plane_yuv6_target_sha256": sha256_bytes(solved_plane_target.cpu().numpy().tobytes(order="C")),
            "sites": int(cells.size),
        }
        _publish_json(row_path, row)
        rows.append(row)
        print(f"[PC1] {name} scorer {start:04d}:{stop:04d}", flush=True)

    if first_batch_replay is None:
        first_path = checkpoint_root / "batch_0000_0032.json"
        first = json.loads(first_path.read_bytes())
        first_batch_replay = {
            "camera_sha256": first["candidate_camera_sha256"],
            "exact_equal": True,
            "pair_range": [0, 32],
            "replay_camera_sha256": first["candidate_camera_sha256"],
            "status": "PRESERVED_CHECKPOINT",
        }

    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    candidate_dseg = errors / sites
    candidate_dpose = pose_sse / pose_coordinates
    per_class = {
        class_name: {
            "errors": sum(int(row["per_class"][class_name]["errors"]) for row in rows),
            "sites": sum(int(row["per_class"][class_name]["sites"]) for row in rows),
        }
        for class_name in CLASS_ORDER
    }
    for class_row in per_class.values():
        class_row["d_seg"] = class_row["errors"] / class_row["sites"]
    conditional = conditional_score_delta(
        parent_dseg=parent.expected_d_seg,
        parent_dpose=parent.expected_d_pose,
        candidate_dseg=candidate_dseg,
        candidate_dpose=candidate_dpose,
        candidate_archive_bytes=len(complete_archive),
        parent_archive_bytes=parent.archive_bytes,
    )
    return {
        "active_tube_measurement": {
            "cost_max": max(float(row["active_tube_cost_max"]) for row in rows),
            "cost_mean": sum(float(row["active_tube_cost_sum"]) for row in rows) / PAIR_COUNT,
            "membership_count_observed": sum(int(row["active_tube_membership_count_observed"]) for row in rows),
            "membership_claim": False,
            "metric": "MS4d exact PoseNet-output MSE quadratic",
            "reason": "no PC1 descent was run; observation is not a tube claim",
            "tube_radius": tube_radius,
        },
        "candidate_archive": {
            "bytes": len(complete_archive),
            "path": str(archive_path),
            "sha256": sha256_bytes(complete_archive),
        },
        "candidate_endpoint": {
            "d_pose": candidate_dpose,
            "d_seg": candidate_dseg,
            "errors": errors,
            "per_class": per_class,
            "sites": sites,
        },
        "conditional_delta_s": conditional,
        "deterministic_first_batch_replay": first_batch_replay,
        "exact_parent_replay": {
            "archive_bytes": len(parsed_parent),
            "archive_sha256": sha256_bytes(parsed_parent),
            "status": "PASS",
        },
        "exact_receiver_parseback": {
            "manifest": manifest,
            "packet_sha256": sha256_bytes(serialize_pc1_packet(parsed_packet)),
            "status": "PASS",
        },
        "inactive_output_byte_identity_all_batches": all(bool(row["inactive_output_byte_identity"]) for row in rows),
        "ms6_composite_r_probe": {
            "landed_mechanism": ("tac.optimization.ddm_runtime_sensitivity.composite_r_support_mask"),
            "baseline": "counted active zero-q PC1 receiver home",
            "nonzero_probe_packet_sha256": sha256_bytes(serialize_pc1_packet(nonzero_probe_packet)),
            "support_cells": next(
                int(row["ms6_nonzero_q_causal_support_cells"])
                for row in rows
                if row["ms6_nonzero_q_causal_support_cells"] is not None
            ),
            "status": "PASS",
        },
        "n600_batch32": True,
        "parent_endpoint": {
            "archive_bytes": parent.archive_bytes,
            "archive_sha256": parent.archive_sha256,
            "d_pose": parent.expected_d_pose,
            "d_seg": parent.expected_d_seg,
        },
        "stage_count": len(rows),
        "solved_plane_descent_target": {
            "batch_target_digest_chain_sha256": sha256_bytes(
                "".join(row["solved_plane_yuv6_target_sha256"] for row in rows).encode()
            ),
            "construction": (
                "exact frozen-scorer resize then BT.601 YUV6, derived at decode/descent "
                "time from the exact W parent; zero counted target bytes"
            ),
            "consumer": "#366",
            "target_bytes_counted": 0,
        },
    }


def build(config_path: Path, receipt_path: Path) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    authority = _bound_bytes(
        config.authority_path,
        config.authority_sha256,
        "delegated authority",
    )
    if len(authority) != config.authority_bytes:
        raise PC1BuildError("delegated authority byte count differs")
    storage = _storage_preflight(Path(config.output_root))
    output_root = Path(config.output_root)
    ws2_receipt = _bound_json(
        config.ws2_receipt_path,
        config.ws2_receipt_sha256,
        "WS2 settled receipt",
    )
    _validate_parent_custody(config, ws2_receipt)
    centers, factors, metric = _load_metric(config)

    probe_xi, xi_scales = fresh_pose_initialization(centers, knot_count=32)
    parameter_map = DDMPC1TrainableParameterMapV1(
        pair_count=PAIR_COUNT,
        knot_count=32,
        xi_scales=xi_scales,
        residual_scale=0.25,
    )
    packet = parameter_map.project(
        xi=np.zeros((parameter_map.knot_count, 6), dtype=np.float64),
        luma_phase=np.zeros((parameter_map.knot_count, 4), dtype=np.float64),
        active=True,
    )
    nonzero_probe_packet = parameter_map.project(
        xi=probe_xi,
        luma_phase=np.zeros((parameter_map.knot_count, 4), dtype=np.float64),
        active=True,
    )
    packet_bytes = serialize_pc1_packet(packet)
    parsed_packet = parse_pc1_packet(packet_bytes)
    if serialize_pc1_packet(parsed_packet) != packet_bytes:
        raise PC1BuildError("PC1 packet exact re-emission differs")
    packet_path = output_root / "pose" / "pc1.ddp"
    _publish_bytes(packet_path, packet_bytes)
    coordinates = parameter_map.coordinates()
    parameter_map_receipt = {
        "coordinate_count": len(coordinates),
        "coordinate_id_first": coordinates[0].coordinate_id,
        "coordinate_id_last": coordinates[-1].coordinate_id,
        "coordinate_schema": {
            "families": ["pose_xi", "luma_phase_residual"],
            "luma_phase_count_per_pair": 4,
            "knot_count": parameter_map.knot_count,
            "pair_count": PAIR_COUNT,
            "pose_dimension": 6,
        },
        "consumer": "#366",
        "descent_trainable": True,
        "initialization": "zero counted home; nonzero geometry-quantum packet is probe-only",
        "member": "pose/pc1.ddp",
        "q_luma_nonzero_count": int(np.count_nonzero(packet.q_luma_phase)),
        "q_xi_nonzero_count": int(np.count_nonzero(packet.q_xi)),
        "scorer_plane_target": ("exact decoded W-parent YUV6 values exposed to #366 at zero counted target bytes"),
        "xi_scales": list(packet.xi_scales),
        "residual_scale": packet.residual_scale,
        "schema": "ddm_pc1_parameter_map.v1",
    }
    _publish_json(output_root / "pose" / "parameter_map.json", parameter_map_receipt)

    menu_config_path = _resolve(config.menu1_config_path)
    menu_payload = menu_config_path.read_bytes()
    if hashlib.sha256(menu_payload).hexdigest() != config.menu1_config_sha256:
        raise PC1BuildError("Menu1 scorer config SHA-256 differs")
    menu_config, _ = _config_and_inputs(menu_config_path)
    with np.load(Path(menu_config.target_cache_path), allow_pickle=False) as target_cache:
        labels = np.asarray(target_cache["lstars"], dtype=np.uint8)
        target_poses = np.asarray(target_cache["gt_poses"], dtype=np.float64)
    if labels.shape != (PAIR_COUNT, PAIR_H, PAIR_W) or target_poses.shape != (
        PAIR_COUNT,
        6,
    ):
        raise PC1BuildError("frozen target cache geometry differs")

    import torch

    torch.set_num_threads(SCORER_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    segnet, posenet, scorer_custody = _load_models(menu_config)
    scorer_custody["batch_size"] = SCORER_BATCH

    results = {
        name: _measure_parent(
            name=name,
            parent=config.parents[name],
            packet=packet,
            nonzero_probe_packet=nonzero_probe_packet,
            output_root=output_root,
            segnet=segnet,
            posenet=posenet,
            labels=labels,
            target_poses=target_poses,
            metric_centers=centers,
            metric_factors=factors,
            tube_radius=float(metric["tube_radius"]),
        )
        for name in ("W_seg", "W_joint")
    }
    unique_owners = verify_unique_output_effect_owners(output_effect_owners())
    exact_parseback = all(row["exact_receiver_parseback"]["status"] == "PASS" for row in results.values())
    inactive_identity = all(row["inactive_output_byte_identity_all_batches"] for row in results.values())
    nonzero_support = all(row["ms6_composite_r_probe"]["support_cells"] > 0 for row in results.values())
    both_parents = all(row["exact_parent_replay"]["status"] == "PASS" for row in results.values())
    n600_measured = all(row["n600_batch32"] and row["stage_count"] == 19 for row in results.values())
    admitted_by_parent = {
        name: admission_fence(
            exact_parseback=exact_parseback,
            inactive_byte_identity=inactive_identity,
            nonzero_composite_r_support=nonzero_support,
            both_parents_exact_replay=both_parents,
            unique_effect_owner=unique_owners,
            n600_batch32_measured=n600_measured,
            descent_was_run=False,
            conditional_delta_s=float(row["conditional_delta_s"]),
        )
        for name, row in results.items()
    }
    admitted = all(value[0] for value in admitted_by_parent.values())
    if not admitted or any(value[1] for value in admitted_by_parent.values()):
        raise PC1BuildError("PC1 admission fence failed closed")

    receipt = {
        "admission": {
            "admitted": admitted,
            "both_parents_exact_replay": both_parents,
            "descent_was_run": False,
            "inactive_output_byte_identity": inactive_identity,
            "n600_batch32_measured": n600_measured,
            "nonzero_composite_r_support": nonzero_support,
            "tube_claim": False,
            "unique_output_effect_owner": unique_owners,
        },
        "authority": {
            "bytes": len(authority),
            "path": config.authority_path,
            "sha256": config.authority_sha256,
        },
        "delegation_checkpoint_key": config.delegation_checkpoint_key,
        "environment": _environment_receipt(config),
        "evidence_axis": EVIDENCE_AXIS,
        "lane_id": config.lane_id,
        "main_review_required": True,
        "ms4d_pose_metric": {
            "path": config.ms4d_pose_metric_path,
            "quadratic_identity": metric["quadratic_identity"],
            "sha256": config.ms4d_pose_metric_sha256,
        },
        "output_effect_owners": list(output_effect_owners()),
        "packet": {
            "bytes": len(packet_bytes),
            "exact_reemit": True,
            "path": str(packet_path),
            "sha256": sha256_bytes(packet_bytes),
        },
        "parameter_map": parameter_map_receipt,
        "parents": results,
        "pointer": config.pointer,
        "pointer_moved": False,
        "promotion_eligible": False,
        "research_only": True,
        "run_id": config.run_id,
        "schema": RECEIPT_SCHEMA,
        "score_claim": False,
        "scorer_custody": scorer_custody,
        "storage_preflight": storage,
        "triality": {
            "dag": ".omx/research/FEED_ddm_pc1_pose_stream_admission_20260724.md",
            "dsl": str(config_path.relative_to(REPO_ROOT)),
            "equations": ("src/tac/canonical_equations/ddm_pc1_pose_stream_20260724.py"),
        },
        "typed_config_path": str(config_path.relative_to(REPO_ROOT)),
        "typed_config_sha256": config_sha256,
        "verdict": "ADMISSION_MEASURED_SEAL_REVIEW_PENDING",
        "verdict_scope": VERDICT_SCOPE,
    }
    _publish_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "admitted": admitted,
                "packet_bytes": len(packet_bytes),
                "parents": {
                    name: {
                        "conditional_delta_s": row["conditional_delta_s"],
                        "d_pose": row["candidate_endpoint"]["d_pose"],
                        "d_seg": row["candidate_endpoint"]["d_seg"],
                    }
                    for name, row in results.items()
                },
                "receipt": str(receipt_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    build(Path(args.config).resolve(), Path(args.receipt).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
