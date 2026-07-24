#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure static ego-hood reassertion over the exact MENU1 winner at n600."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_hood_static_reassert import (  # noqa: E402
    class_transition_rows,
    decode_stored_support,
    derive_hood_supports,
    encode_stored_support,
    expand_support_to_camera,
    reassert_frame1,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)

SCHEMA = "ddm_mc1_hood_static_reassert_measurement.v1"
CONFIG_SCHEMA = "DDMMC1HoodStaticReassertConfigV1"
LANE_ID = "lane_ddm_mc1_hood_static_reassert_20260724"
DELEGATION_KEY = "codex_delegate:ddm_mc1_hood_static_reassert:20260724T003346Z"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
PARENT_ID = "statistics_hard_analytic_composed_frame1"
BASE_ID = "v19c_base"


class MC1Error(RuntimeError):
    """Fail-closed MC1 measurement error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise MC1Error(f"regular non-symlink file required: {path}")
    return path.read_bytes()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _bound(path: str, expected: str, label: str) -> bytes:
    payload = _read_regular(_resolve(path))
    actual = sha256_bytes(payload)
    if actual != expected:
        raise MC1Error(f"{label} sha256 differs: {actual} != {expected}")
    return payload


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path) != payload:
            raise MC1Error(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _publish_json(path: Path, value: Any) -> None:
    _publish(path, _canonical(value))


def _import_menu1() -> Any:
    source = REPO_ROOT / "tools" / "measure_ddm_menu1_realized_flip_menu.py"
    spec = importlib.util.spec_from_file_location("_ddm_menu1_mc1_parent", source)
    if spec is None or spec.loader is None:
        raise MC1Error("MENU1 measurement module import spec is absent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MC1Config(BaseModel):
    """Strict local n600 authority contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMMC1HoodStaticReassertConfigV1"] = Field(
        CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: str = Field(min_length=12)
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    menu1_config_path: str
    menu1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    menu1_receipt_path: str
    menu1_receipt_sha256: Literal[
        "2fc12eb505aa7de140b5e785e5fb528c349fd72b423a1821f23b70ea21d6f29d"
    ]
    checkpoint_root: str
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"
    execution_allowed: Literal[True] = True
    paid_dispatch_allowed: Literal[False] = False
    exact_eval_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    training_allowed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _paths_and_tier(self) -> MC1Config:
        if not Path(self.checkpoint_root).is_absolute():
            raise ValueError("checkpoint_root must be absolute")
        if not self.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/"):
            raise ValueError("checkpoint_root must use the primary SSD tier")
        return self

    def stable_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )


def _checkpoint_paths(
    root: Path, candidate_id: str, start: int, stop: int
) -> tuple[Path, Path]:
    stage = root / "02_measurements" / candidate_id
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _menu_checkpoint_paths(
    root: Path, candidate_id: str, start: int, stop: int
) -> tuple[Path, Path]:
    stage = root / "02_measurements" / candidate_id
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _load_all_base_cells(menu_root: Path, *, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, 600, batch_size):
        stop = min(start + batch_size, 600)
        _row, arrays = _menu_checkpoint_paths(menu_root, BASE_ID, start, stop)
        with np.load(arrays, allow_pickle=False) as stored:
            rows.append(np.asarray(stored["cells"], dtype=np.uint8))
    result = np.ascontiguousarray(np.concatenate(rows, axis=0))
    if result.shape[0] != 600:
        raise MC1Error("MENU1 base checkpoint coverage differs")
    return result


def _save_arrays(path: Path, *, cells: np.ndarray, pose6: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["cells"], cells) or not np.array_equal(
                stored["pose6"], pose6
            ):
                raise MC1Error(f"immutable array checkpoint differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, cells=cells, pose6=pose6)
    os.replace(temporary, path)


def _pose_input_delta(posenet: Any, before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    import torch

    def _tensor(camera: np.ndarray) -> Any:
        return (
            torch.from_numpy(np.ascontiguousarray(camera))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .float()
        )

    with torch.inference_mode():
        a = posenet.preprocess_input(_tensor(before)).cpu()
        b = posenet.preprocess_input(_tensor(after)).cpu()
    delta = (b - a).abs()
    return {
        "official_posenet_input_changed_coordinates": int(torch.count_nonzero(delta)),
        "official_posenet_input_l1": float(delta.sum(dtype=torch.float64)),
        "official_posenet_input_linf": float(delta.max()),
        "official_posenet_input_coordinates": int(delta.numel()),
    }


def _support_for_batch(
    *,
    kind: str,
    start: int,
    stop: int,
    supports: Any,
    semantic: np.ndarray,
    owned: np.ndarray,
) -> np.ndarray:
    if kind == "static_stored":
        return np.broadcast_to(supports.static, (stop - start, *supports.static.shape))
    if kind == "per_frame_stored":
        return supports.per_frame[start:stop]
    if kind == "decoder_semantic_free":
        return np.ascontiguousarray((semantic == supports.hood_class) & owned)
    raise MC1Error(f"unknown support kind: {kind}")


def _sum_nested_class_rows(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    names = tuple(rows[0]["per_class"])
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        sites = sum(int(row["per_class"][name]["sites"]) for row in rows)
        result[name] = {
            key: sum(int(row["per_class"][name][key]) for row in rows)
            for key in (
                "sites",
                "errors_before",
                "errors_after",
                "errors_corrected",
                "errors_introduced",
                "delta_errors_realized",
            )
        }
        result[name]["d_seg_after"] = result[name]["errors_after"] / sites
    return result


def _measure(
    *,
    config: MC1Config,
    menu1: Any,
    menu_config: Any,
    receiver: Any,
    palette: np.ndarray,
    statistics_payload: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    supports: Any,
    support_kind: str,
    support_counted_bytes: int,
    root: Path,
    menu_root: Path,
    parent_archive_bytes: int,
) -> dict[str, Any]:
    candidate_id = f"mc1_hood_reassert_{support_kind}_frame1"
    rows: list[dict[str, Any]] = []
    for start in range(0, 600, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, 600)
        row_path, array_path = _checkpoint_paths(root, candidate_id, start, stop)
        if row_path.exists() and array_path.exists():
            row = json.loads(_read_regular(row_path))
            if (
                row.get("typed_config_sha256") != config.stable_hash()
                or row.get("candidate_id") != candidate_id
            ):
                raise MC1Error("measurement resume identity differs")
            rows.append(row)
            continue
        ids = tuple(range(start, stop))
        base_camera = receiver.render_camera_pairs(ids)
        semantic, owned = menu1._semantic_cells(receiver, ids, base_camera, palette)
        winner = menu1._geometry_statistics_camera(
            base_camera=base_camera,
            semantic=semantic,
            owned=owned,
            palette=palette,
            statistics_payload=statistics_payload,
        )
        parent_row_path, parent_arrays_path = _menu_checkpoint_paths(
            menu_root, PARENT_ID, start, stop
        )
        parent_row = json.loads(_read_regular(parent_row_path))
        if sha256_bytes(winner.tobytes()) != parent_row["camera_sha256"]:
            raise MC1Error("fresh MENU1 winner camera differs from preserved checkpoint")
        scorer_support = _support_for_batch(
            kind=support_kind,
            start=start,
            stop=stop,
            supports=supports,
            semantic=semantic,
            owned=owned,
        )
        camera_support = expand_support_to_camera(
            scorer_support,
            batch_size=stop - start,
            camera_hw=menu1.CAMERA_HW,
        )
        camera = reassert_frame1(
            winner_camera=winner,
            base_camera=base_camera,
            camera_support=camera_support,
        )
        base_control = reassert_frame1(
            winner_camera=base_camera,
            base_camera=base_camera,
            camera_support=camera_support,
        )
        if not np.array_equal(base_control, base_camera):
            raise MC1Error("base-only reassert control was not byte-identical")
        cells, pose6 = menu1._forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = menu1._forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(
                pose6, replay_pose6
            ):
                raise MC1Error("first-batch deterministic replay differs")
        with np.load(parent_arrays_path, allow_pickle=False) as stored:
            parent_cells = np.asarray(stored["cells"], dtype=np.uint8)
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        per_class = class_transition_rows(
            before=parent_cells,
            after=cells,
            target=target,
            class_names=menu1.CLASS_NAMES,
        )
        transition = menu1.transition_counts(
            before=parent_cells, after=cells, target=target
        )
        pose_input = _pose_input_delta(posenet, winner, camera)
        changed = np.any(camera[:, 1] != winner[:, 1], axis=-1)
        row = {
            "schema": "ddm_mc1_hood_static_reassert_batch.v1",
            "typed_config_sha256": config.stable_hash(),
            "candidate_id": candidate_id,
            "support_kind": support_kind,
            "pair_range": [start, stop],
            "errors": int(np.count_nonzero(cells != target)),
            "sites": int(cells.size),
            "pose_squared_error_sum": float(
                np.square(pose6 - target_pose).sum(dtype=np.float64)
            ),
            "pose_coordinates": int(pose6.size),
            "transition_from_parent": transition,
            "per_class": per_class,
            "support_cells": int(np.count_nonzero(scorer_support)),
            "support_sites": int(scorer_support.size),
            "camera_pixels_changed_vs_parent": int(np.count_nonzero(changed)),
            "frame0_byte_identical": bool(np.array_equal(camera[:, 0], base_camera[:, 0])),
            "base_only_control_byte_identical": True,
            **pose_input,
            "camera_sha256": sha256_bytes(camera.tobytes()),
            "cells_sha256": sha256_bytes(cells.tobytes()),
            "pose6_sha256": sha256_bytes(pose6.tobytes()),
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        }
        _save_arrays(array_path, cells=cells, pose6=pose6)
        _publish_json(row_path, row)
        rows.append(row)
        print(f"[MC1] {support_kind} {start:04d}:{stop:04d}", flush=True)
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_pose = pose_sse / pose_coordinates
    archive_bytes = parent_archive_bytes + support_counted_bytes
    per_class = _sum_nested_class_rows(rows)
    transition = {
        key: sum(int(row["transition_from_parent"][key]) for row in rows)
        for key in rows[0]["transition_from_parent"]
    }
    return {
        "candidate_id": candidate_id,
        "composition_pool_id": "static_reassert",
        "parent_candidate_id": PARENT_ID,
        "support_kind": support_kind,
        "archive_bytes": archive_bytes,
        "delta_counted_bytes_vs_parent": support_counted_bytes,
        "byte_partition": {
            "COUNTED": support_counted_bytes,
            "FREE": 0,
            "NULL": 0,
            "FREE_source": (
                None
                if support_counted_bytes
                else "decoder-derived from counted V19C semantic geometry"
            ),
        },
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "d_pose": d_pose,
        "advisory_objective": menu1.advisory_objective(
            errors=errors, sites=sites, d_pose=d_pose, bytes_=archive_bytes
        ),
        "transition_from_parent": transition,
        "per_class": per_class,
        "support_cells": sum(int(row["support_cells"]) for row in rows),
        "support_sites": sum(int(row["support_sites"]) for row in rows),
        "camera_pixels_changed_vs_parent": sum(
            int(row["camera_pixels_changed_vs_parent"]) for row in rows
        ),
        "official_posenet_input_changed_coordinates": sum(
            int(row["official_posenet_input_changed_coordinates"]) for row in rows
        ),
        "official_posenet_input_l1": sum(
            float(row["official_posenet_input_l1"]) for row in rows
        ),
        "official_posenet_input_linf": max(
            float(row["official_posenet_input_linf"]) for row in rows
        ),
        "frame0_byte_identical": all(row["frame0_byte_identical"] for row in rows),
        "base_only_control_byte_identical": all(
            row["base_only_control_byte_identical"] for row in rows
        ),
        "batch_count": len(rows),
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": sha256_bytes(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in rows).encode()
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def run(config_path: Path, output_directory: Path) -> Path:
    config = MC1Config.model_validate_json(_read_regular(config_path))
    menu1 = _import_menu1()
    menu_config_payload = _bound(
        config.menu1_config_path, config.menu1_config_sha256, "MENU1 config"
    )
    menu_receipt = json.loads(
        _bound(config.menu1_receipt_path, config.menu1_receipt_sha256, "MENU1 receipt")
    )
    if (
        menu_receipt.get("verdict") != "MENU1_MEASURED_BOX_NOT_REACHED"
        or menu_receipt.get("score_claim") is not False
    ):
        raise MC1Error("MENU1 authority receipt differs")
    menu_config = menu1.Menu1Config.model_validate_json(menu_config_payload)
    _menu_config, inputs = menu1._config_and_inputs(_resolve(config.menu1_config_path))
    if menu_config != _menu_config:
        raise MC1Error("MENU1 typed config paths disagree")
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = 512 << 20
    if free < required:
        raise MC1Error("SSD storage preflight failed")
    menu_root = Path(menu_config.checkpoint_root)
    base_cells = _load_all_base_cells(
        menu_root, batch_size=config.scorer_batch_size
    )
    supports = derive_hood_supports(base_cells)
    static_payload = encode_stored_support(supports.static)
    per_frame_payload = encode_stored_support(supports.per_frame)
    if not np.array_equal(decode_stored_support(static_payload), supports.static):
        raise MC1Error("static support parse-back differs")
    if not np.array_equal(
        decode_stored_support(per_frame_payload), supports.per_frame
    ):
        raise MC1Error("per-frame support parse-back differs")
    support_dir = root / "01_support"
    static_path = support_dir / "static_majority_support.bin"
    per_frame_path = support_dir / "per_frame_support.bin"
    _publish(static_path, static_payload)
    _publish(per_frame_path, per_frame_payload)
    receiver = receive_preuint8_q8_archive(inputs["archive"])
    palette = menu1._palette(receiver)
    statistics_payload = _read_regular(menu_root / "01_local_statistics_payload.bin")
    if sha256_bytes(statistics_payload) != menu_receipt["local_statistics_payload"][
        "payload_sha256"
    ]:
        raise MC1Error("MENU1 statistics payload differs")
    labels = open_stored_npy_memmap(Path(menu_config.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(menu_config.target_cache_path), "gt_poses")
    segnet, posenet, scorer_custody = menu1._load_models(menu_config)
    parent = next(
        row
        for row in menu_receipt["measured_menu_rows"]
        if row["row_id"] == PARENT_ID
    )
    candidates = []
    for support_kind, counted in (
        ("static_stored", len(static_payload)),
        ("per_frame_stored", len(per_frame_payload)),
        ("decoder_semantic_free", 0),
    ):
        candidates.append(
            _measure(
                config=config,
                menu1=menu1,
                menu_config=menu_config,
                receiver=receiver,
                palette=palette,
                statistics_payload=statistics_payload,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
                supports=supports,
                support_kind=support_kind,
                support_counted_bytes=counted,
                root=root,
                menu_root=menu_root,
                parent_archive_bytes=int(parent["archive_bytes"]),
            )
        )
    best = min(candidates, key=lambda row: row["advisory_objective"])
    parent_s = menu1.advisory_objective(
        errors=round(float(parent["d_seg"]) * 117_964_800),
        sites=117_964_800,
        d_pose=float(parent["d_pose"]),
        bytes_=int(parent["archive_bytes"]),
    )
    positive = best["advisory_objective"] < parent_s
    for row in candidates:
        row["delta_advisory_objective_vs_parent"] = (
            row["advisory_objective"] - parent_s
        )
        row["admitted_within_static_reassert_pool"] = row is best and positive
        counted = int(row["delta_counted_bytes_vs_parent"])
        improvement = parent_s - row["advisory_objective"]
        row["joint_score_units_improved_per_counted_byte"] = (
            improvement / counted if counted else None
        )
    base_row = menu_receipt["curve"][0]
    residual_mycar = int(best["per_class"]["MyCar"]["errors_after"])
    receipt = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.stable_hash(),
        "input_custody": {
            "menu1_config_path": config.menu1_config_path,
            "menu1_config_sha256": config.menu1_config_sha256,
            "menu1_receipt_path": config.menu1_receipt_path,
            "menu1_receipt_sha256": config.menu1_receipt_sha256,
            "menu1_parent_candidate_id": PARENT_ID,
            "menu1_parent": parent,
            "v19c_total_errors": 2_923_991,
            "v19c_residual_bucket_errors": 2_265_811,
        },
        "support_derivation": {
            "method": "identify_static_hood_class on exact V19C base argmax; no hard-coded class index",
            "detected_class_id": supports.hood_class,
            "class_evidence": list(supports.evidence),
            "static_mean_frame_iou": supports.static_mean_frame_iou,
            "static_min_frame_iou": supports.static_min_frame_iou,
            "static_support_cells": int(np.count_nonzero(supports.static)),
            "per_frame_support_cells": int(np.count_nonzero(supports.per_frame)),
            "partition": {
                "static_stored": {
                    "COUNTED": len(static_payload),
                    "FREE": 0,
                    "NULL": 0,
                    "path": str(static_path),
                    "sha256": sha256_bytes(static_payload),
                    "parse_back_identical": True,
                },
                "per_frame_stored": {
                    "COUNTED": len(per_frame_payload),
                    "FREE": 0,
                    "NULL": 0,
                    "path": str(per_frame_path),
                    "sha256": sha256_bytes(per_frame_payload),
                    "parse_back_identical": True,
                },
                "decoder_semantic_free": {
                    "COUNTED": 0,
                    "FREE": 0,
                    "NULL": 0,
                    "FREE_source": (
                        "support re-expanded from already-counted V19C semantic geometry"
                    ),
                    "scorer_weights_required_at_decode": False,
                },
            },
        },
        "composition": {
            "order": "MENU1 frame1 paint winner -> restore V19C frame1 bytes on hood support",
            "frame0_byte_identical_required": True,
            "composition_pool_id": "static_reassert",
            "paint_support_disjoint": False,
            "interaction_status": "MEASURED_EXACT_ORDERED_COMPOSITION_NOT_ASSUMED_ADDITIVE",
        },
        "candidates": candidates,
        "pool_winner": best,
        "pool_positive": positive,
        "base_v19c_control": {
            "candidate_id": BASE_ID,
            "operation": "restore V19C bytes onto identical V19C bytes on the same support",
            "byte_identical": bool(best["base_only_control_byte_identical"]),
            "measurement_reuse": (
                "exact byte identity permits reuse of the same n600 frozen-scorer row"
            ),
            "errors": int(base_row["errors"]),
            "d_seg": float(base_row["d_seg"]),
            "d_pose": float(base_row["d_pose"]),
            "MyCar_errors": int(base_row["per_class"]["MyCar"]["errors"]),
            "shrinks_base_mycar_bucket": False,
        },
        "waterfill_route": {
            "pool_id": "static_reassert",
            "admit": positive,
            "measured_delta_S": best["advisory_objective"] - parent_s,
            "measured_delta_bytes": best["delta_counted_bytes_vs_parent"],
            "same_pool_law": "select one support formulation; never sum their deltas",
            "rs1_366_mycar_residual_errors": residual_mycar,
            "rs1_366_scope": (
                "remaining MyCar residual after static reassert; no family closure"
            ),
        },
        "verdict": (
            "MC1_MEASURED_POSITIVE_STATIC_REASSERT"
            if positive
            else "MC1_MEASURED_INSTANCE_NOT_JOINT_POSITIVE"
        ),
        "verdict_scope": (
            "INSTANCE: exact V19C base bytes reasserted after the MENU1 "
            "local-statistics+hard-placement+analytic-coverage frame1 winner; "
            "static-field and broader reassert formulations remain open"
        ),
        "scorer_custody": scorer_custody,
        "storage_preflight": {
            "tier": "/Volumes/VertigoDataTier/pact",
            "checkpoint_root": str(root),
            "required_free_bytes": required,
            "observed_free_bytes_at_least": required,
            "status": "PASS",
            "cleanup_policy": "preserve immutable n600 checkpoints and support payloads",
        },
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": config.pointer,
        "pointer_moved": False,
        "paid_dispatch": False,
        "exact_eval": False,
        "training": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "ddm_mc1_hood_static_reassert_receipt.json"
    _publish_json(output, receipt)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = run(_resolve(args.config), _resolve(args.output_directory))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
