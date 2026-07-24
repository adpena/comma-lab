#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize and freshly remeasure both receiver-closed WS1 warm starts.

This bounded local-only producer writes exact archive bytes and immutable
batch-32 frozen-scorer checkpoints to the primary SSD. It cannot train,
dispatch, run contest evaluation, promote, or move the frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_realized_flip_menu import (  # noqa: E402
    EVIDENCE_AXIS,
    SEG_HW,
    advisory_objective,
    sha256_bytes,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    W_JOINT,
    W_SEG,
    compile_ws1_warm_start_archive,
    receive_ws1_warm_start_archive,
)
from tac.optimization.direct_description_carrier_compose import CLASS_ORDER  # noqa: E402
from tac.optimization.direct_description_joint_descent import lift_v15_archive  # noqa: E402
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    _config_and_inputs,
    _forward,
    _load_models,
)

SCHEMA = "ddm_ws2_warm_start_custody_producer.v1"
CONFIG_SCHEMA = "DDMWS2WarmStartCustodyProducerConfigV1"
LANE_ID = "lane_ddm_ws2_warm_start_custody_producer_20260724"
DELEGATION_KEY = "codex_delegate:ddm_ws2_warm_start_custody_producer:20260724T053455Z"
POINTER = "0.1910828242 [contest-CPU]"
DEFAULT_CONFIG = (
    REPO_ROOT
    / ".omx/research/configs/ddm_ws2_warm_start_custody_producer_20260724.json"
)
DEFAULT_RECEIPT = (
    REPO_ROOT
    / ".omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json"
)


class CustodyError(RuntimeError):
    """Fail-closed custody or exact-measurement error."""


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_archive_path: str
    base_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_archive_bytes: Literal[137827] = 137_827
    payload_path: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_bytes: int = Field(gt=0)
    expected_archive_bytes: int = Field(gt=0)
    expected_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_d_seg: float = Field(ge=0.0)
    expected_d_pose: float = Field(ge=0.0)


class CustodyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMWS2WarmStartCustodyProducerConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    authority_path: str
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_bytes: Literal[8479] = 8479
    menu1_config_path: str
    menu1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ws1_receipt_path: str
    ws1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[32] = 32
    scorer_threads: Literal[4] = 4
    candidates: dict[Literal["W_seg", "W_joint"], CandidateConfig]
    execution_allowed: Literal[True] = True
    paid_dispatch_allowed: Literal[False] = False
    exact_contest_eval_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed(self) -> CustodyConfig:
        if set(self.candidates) != {W_SEG, W_JOINT}:
            raise ValueError("exactly W_seg and W_joint candidates are required")
        if not self.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/"):
            raise ValueError("checkpoint_root must use primary SSD")
        for name, row in self.candidates.items():
            if row.base_archive_bytes + row.payload_bytes != row.expected_archive_bytes:
                raise ValueError(f"{name} byte accounting differs")
        return self

    def stable_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CustodyError(f"regular non-symlink file required: {path}")
    return path.read_bytes()


def _bound(
    path_value: str,
    expected_sha: str,
    expected_bytes: int | None,
    label: str,
) -> bytes:
    path = _resolve(path_value)
    payload = _read(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise CustodyError(f"{label} SHA-256 differs")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise CustodyError(f"{label} byte count differs")
    return payload


def _canonical(value: Any) -> bytes:
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


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != payload:
            raise CustodyError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _publish_npz(path: Path, *, cells: np.ndarray, pose6: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["cells"], cells) or not np.array_equal(
                stored["pose6"], pose6
            ):
                raise CustodyError(f"immutable arrays differ: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, cells=cells, pose6=pose6)
    os.replace(temporary, path)


def _storage_preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = 1 << 30
    if free < required:
        raise CustodyError(
            f"storage preflight needs {required} free bytes, found {free}"
        )
    return {
        "status": "PASS",
        "tier": "/Volumes/VertigoDataTier/pact",
        "checkpoint_root": str(root),
        "free_bytes": free,
        "required_free_bytes": required,
        "cleanup": "immutable batch checkpoints retained; no scratch survives success",
    }


def _batch_row(
    *,
    config: CustodyConfig,
    name: str,
    archive: bytes,
    start: int,
    stop: int,
    camera: np.ndarray,
    cells: np.ndarray,
    pose6: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(labels[start:stop], dtype=np.uint8)
    target_pose = np.asarray(poses[start:stop], dtype=np.float64)
    per_class = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        mask = target == class_id
        per_class[class_name] = {
            "errors": int(np.count_nonzero((cells != target) & mask)),
            "sites": int(np.count_nonzero(mask)),
        }
    return {
        "schema": "ddm_ws2_warm_start_batch32.v1",
        "typed_config_sha256": config.stable_hash(),
        "candidate": name,
        "pair_range": [start, stop],
        "errors": int(np.count_nonzero(cells != target)),
        "sites": int(cells.size),
        "pose_squared_error_sum": float(
            np.square(pose6 - target_pose).sum(dtype=np.float64)
        ),
        "pose_coordinates": int(pose6.size),
        "per_class": per_class,
        "camera_sha256": sha256_bytes(camera.tobytes()),
        "cells_sha256": sha256_bytes(cells.tobytes()),
        "pose6_sha256": sha256_bytes(pose6.tobytes()),
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "scorer_batch_size": 32,
        "scorer_threads": 4,
        "camera_batch_released_after_forward": True,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _aggregate(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    archive: bytes,
) -> dict[str, Any]:
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    per_class = {
        class_name: {
            "errors": sum(
                int(row["per_class"][class_name]["errors"]) for row in rows
            ),
            "sites": sum(
                int(row["per_class"][class_name]["sites"]) for row in rows
            ),
        }
        for class_name in CLASS_ORDER
    }
    for value in per_class.values():
        value["d_seg"] = value["errors"] / value["sites"]
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    return {
        "candidate": name,
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "advisory_objective": advisory_objective(
            errors=errors,
            sites=sites,
            d_pose=d_pose,
            bytes_=len(archive),
        ),
        "per_class": per_class,
        "batch_count": len(rows),
        "scorer_batch_size": 32,
        "batch_digest_chain_sha256": sha256_bytes(
            b"".join(_canonical(dict(row)) for row in rows)
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def produce(config_path: Path, receipt_path: Path) -> dict[str, Any]:
    config = CustodyConfig.model_validate_json(_read(config_path))
    _bound(
        config.authority_path,
        config.authority_sha256,
        config.authority_bytes,
        "authority",
    )
    menu_bytes = _bound(
        config.menu1_config_path,
        config.menu1_config_sha256,
        None,
        "MENU1 config",
    )
    ws1_bytes = _bound(
        config.ws1_receipt_path,
        config.ws1_receipt_sha256,
        None,
        "WS1 receipt",
    )
    source_receipt = json.loads(ws1_bytes)
    root = Path(config.checkpoint_root)
    storage = _storage_preflight(root)
    menu_config, _ = _config_and_inputs(_resolve(config.menu1_config_path))
    if menu_config.pair_count != 600 or menu_config.scorer_threads != 4:
        raise CustodyError("MENU1 target/scorer geometry differs")
    labels = open_stored_npy_memmap(
        Path(menu_config.target_cache_path), "lstars"
    )
    poses = open_stored_npy_memmap(
        Path(menu_config.target_cache_path), "gt_poses"
    )
    if labels.shape != (600, *SEG_HW) or poses.shape != (600, 6):
        raise CustodyError("target cache geometry differs")

    archives: dict[str, bytes] = {}
    archive_custody: dict[str, Any] = {}
    for name in (W_SEG, W_JOINT):
        row = config.candidates[name]
        base = _bound(
            row.base_archive_path,
            row.base_archive_sha256,
            row.base_archive_bytes,
            f"{name} base",
        )
        payload = _bound(
            row.payload_path,
            row.payload_sha256,
            row.payload_bytes,
            f"{name} payload",
        )
        archive = compile_ws1_warm_start_archive(
            base,
            candidate=name,
            payload=payload,
        )
        if (
            len(archive) != row.expected_archive_bytes
            or sha256_bytes(archive) != row.expected_archive_sha256
        ):
            raise CustodyError(f"{name} materialized archive identity differs")
        receiver = receive_ws1_warm_start_archive(archive)
        if receiver.parsed.exact_reemit() != archive:
            raise CustodyError(f"{name} receiver parse/re-emit differs")
        lift = lift_v15_archive(archive)
        if lift.exact_reemit() != archive or len(lift.parameter_names) != 368:
            raise CustodyError(f"{name} J5 lift/recompile custody differs")
        archive_path = (
            root / "01_archives" / f"{name}.zip.receipt-bytes"
        )
        _publish(archive_path, archive)
        archives[name] = archive
        archive_custody[name] = {
            **dict(receiver.parsed.custody),
            "archive_path": str(archive_path),
            "archive_path_sha256": hashlib.sha256(_read(archive_path)).hexdigest(),
            "receiver_parse_reemit_byte_identical": True,
            "j5_low_dim_parameter_count": 368,
            "j5_stage00_lift_recompile_byte_identical": True,
        }

    segnet, posenet, scorer_custody = _load_models(menu_config)
    scorer_custody["batch_size"] = 32
    rows_by_name: dict[str, list[dict[str, Any]]] = {
        W_SEG: [],
        W_JOINT: [],
    }
    deterministic_replay: dict[str, Any] = {}
    receivers = {
        name: receive_ws1_warm_start_archive(value)
        for name, value in archives.items()
    }
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        ids = tuple(range(start, stop))
        for name in (W_SEG, W_JOINT):
            stage = root / "02_batch32_measurements" / name
            row_path = stage / f"batch_{start:04d}_{stop:04d}.json"
            array_path = stage / f"batch_{start:04d}_{stop:04d}.npz"
            print(
                f"[WS2] {name} receiver/scorer {start:04d}:{stop:04d}",
                flush=True,
            )
            camera = receivers[name].render_camera_pairs(ids)
            if row_path.exists() and array_path.exists():
                existing = json.loads(_read(row_path))
                with np.load(array_path, allow_pickle=False) as stored:
                    stored_cells = np.asarray(stored["cells"], dtype=np.uint8)
                    stored_pose6 = np.asarray(stored["pose6"], dtype=np.float64)
                if (
                    existing.get("typed_config_sha256") != config.stable_hash()
                    or existing.get("camera_sha256")
                    != sha256_bytes(camera.tobytes())
                    or existing.get("archive_sha256")
                    != sha256_bytes(archives[name])
                    or existing.get("cells_sha256")
                    != sha256_bytes(stored_cells.tobytes())
                    or existing.get("pose6_sha256")
                    != sha256_bytes(stored_pose6.tobytes())
                ):
                    raise CustodyError(
                        f"{name} resume checkpoint identity differs"
                    )
                if start == 0:
                    replay_cells, replay_pose6 = _forward(
                        segnet,
                        posenet,
                        camera,
                    )
                    replay_cells_again, replay_pose6_again = _forward(
                        segnet,
                        posenet,
                        camera,
                    )
                    if (
                        not np.array_equal(stored_cells, replay_cells)
                        or not np.array_equal(stored_pose6, replay_pose6)
                        or not np.array_equal(replay_cells, replay_cells_again)
                        or not np.array_equal(replay_pose6, replay_pose6_again)
                    ):
                        raise CustodyError(
                            f"{name} resumed first-batch replay differs"
                        )
                    deterministic_replay[name] = {
                        "pair_range": [0, 32],
                        "camera_sha256": sha256_bytes(camera.tobytes()),
                        "cells_sha256": sha256_bytes(stored_cells.tobytes()),
                        "pose6_sha256": sha256_bytes(stored_pose6.tobytes()),
                        "status": "PASS",
                        "resume_replay": True,
                    }
                rows_by_name[name].append(existing)
                continue
            cells, pose6 = _forward(segnet, posenet, camera)
            if start == 0:
                cells_again, pose_again = _forward(
                    segnet,
                    posenet,
                    camera,
                )
                if not np.array_equal(
                    cells, cells_again
                ) or not np.array_equal(pose6, pose_again):
                    raise CustodyError(
                        f"{name} first batch deterministic replay differs"
                    )
                deterministic_replay[name] = {
                    "pair_range": [0, 32],
                    "camera_sha256": sha256_bytes(camera.tobytes()),
                    "cells_sha256": sha256_bytes(cells.tobytes()),
                    "pose6_sha256": sha256_bytes(pose6.tobytes()),
                    "status": "PASS",
                }
            row = _batch_row(
                config=config,
                name=name,
                archive=archives[name],
                start=start,
                stop=stop,
                camera=camera,
                cells=cells,
                pose6=pose6,
                labels=labels,
                poses=poses,
            )
            _publish_npz(array_path, cells=cells, pose6=pose6)
            _publish(row_path, _canonical(row))
            rows_by_name[name].append(row)

    endpoints = {
        name: _aggregate(name, rows_by_name[name], archives[name])
        for name in (W_SEG, W_JOINT)
    }
    sealed_endpoint_comparison = {}
    for name, endpoint in endpoints.items():
        expected = config.candidates[name]
        if endpoint["d_seg"] != expected.expected_d_seg:
            raise CustodyError(
                f"{name} exact d_seg differs: "
                f"{endpoint['d_seg']} != {expected.expected_d_seg}"
            )
        pose_delta = endpoint["d_pose"] - expected.expected_d_pose
        if abs(pose_delta) > 1e-7:
            raise CustodyError(
                f"{name} d_pose differs: "
                f"{endpoint['d_pose']} != {expected.expected_d_pose}"
            )
        sealed_endpoint_comparison[name] = {
            "sealed_batch16_d_seg": expected.expected_d_seg,
            "fresh_batch32_d_seg": endpoint["d_seg"],
            "d_seg_exact_equal": True,
            "sealed_batch16_d_pose": expected.expected_d_pose,
            "fresh_batch32_d_pose": endpoint["d_pose"],
            "d_pose_delta_from_reduction_regrouping": pose_delta,
            "absolute_d_pose_tolerance": 1e-7,
            "status": "PASS",
        }

    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "typed_config_path": str(config_path.relative_to(REPO_ROOT)),
        "typed_config_sha256": config.stable_hash(),
        "authority": {
            "path": config.authority_path,
            "sha256": config.authority_sha256,
            "bytes": config.authority_bytes,
        },
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "storage_preflight": storage,
        "source_ws1_receipt": {
            "path": config.ws1_receipt_path,
            "sha256": config.ws1_receipt_sha256,
            "verdict": source_receipt.get("verdict"),
        },
        "menu1_config_sha256": hashlib.sha256(menu_bytes).hexdigest(),
        "archive_custody": archive_custody,
        "fresh_batch32_deterministic_replay": deterministic_replay,
        "fresh_batch32_endpoints": endpoints,
        "sealed_batch16_endpoint_comparison": sealed_endpoint_comparison,
        "scorer_custody": scorer_custody,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "research_only": True,
        "promotion_eligible": False,
        "main_review_required": True,
        "verdict": (
            "BOTH_WS1_RECEIVER_CLOSED_ARCHIVES_MATERIALIZED_AND_"
            "EXACT_BATCH32_REMEASURED"
        ),
        "verdict_scope": (
            "[macOS-CPU frozen-scorer advisory]; no contest eval, training, "
            "dispatch, promotion, or frontier mutation"
        ),
    }
    _publish(receipt_path, _canonical(receipt))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    receipt = produce(args.config.resolve(), args.receipt.resolve())
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
