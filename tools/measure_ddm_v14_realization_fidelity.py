#!/usr/bin/env python3
"""Measure the counted V14 camera-placement repair over the V13 G1 receiver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    EVIDENCE_AXIS,
    REALIZATION_PAINT_ORDER,
    WORLDSHEET_G1_MEMBER,
    ReceiverRealizationProfileV1,
    _decode_lane_knots,
    _decode_lane_programs,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    CLASS_NAMES,
    _storage_preflight,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    POINTER_SCORE_TEXT,
    DirectDescriptionError,
    _read_regular_file_once,
    _sha256,
    rfc8785_canonicalize,
)
from tac.through_r.resolution_chain import render_grid_to_camera_uint8  # noqa: E402

RESULT_SCHEMA = "ddm_v14_realization_fidelity_receipt.v1"
PAIR_SHAPE = (384, 512)
G4_RECEIPT_PATH = (
    ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/"
    "ddm_g4_spatial_stationarity_receipt.json"
)
G4_RECEIPT_SHA256 = "bea555b95aeaa11f4209df5333010c41c5495dd789def2a4f7a2a91973f3408c"


class DDMV14RealizationFidelityConfigV1(BaseModel):
    """Typed local-only config; one invocation advances at most one candidate stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DDMV14RealizationFidelityConfigV1"] = Field(
        default="DDMV14RealizationFidelityConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    seed: Literal[1234] = 1234
    pair_start: Literal[0, 448]
    pair_count: Literal[64, 600]
    v13_receipt_path: StrictStr
    v13_receipt_sha256: StrictStr
    lane_phase_receipt_path: StrictStr
    lane_phase_receipt_sha256: StrictStr
    target_cache_path: StrictStr
    target_cache_bytes: StrictInt
    target_cache_sha256: StrictStr
    upstream_root: StrictStr
    scorer_threads: StrictInt = Field(ge=1, le=16)
    scorer_batch_size: Literal[16] = 16
    movable_prototype_rgb_u8: tuple[StrictInt, StrictInt, StrictInt] = (107, 0, 114)
    candidate_ladder: tuple[Literal["islands", "both"], ...] = ("islands", "both")
    max_candidate_stages_per_invocation: Literal[1] = 1
    dseg_gate: Literal[0.00116] = 0.00116
    archive_box_bytes: Literal[200000] = 200000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMV14RealizationFidelityConfigV1:
        if (self.pair_start, self.pair_count) not in {(448, 64), (0, 600)}:
            raise ValueError("v14 windows must be n64 [448,512) or full n600")
        for name in (
            "v13_receipt_sha256",
            "lane_phase_receipt_sha256",
            "target_cache_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if any(isinstance(value, bool) or not 0 <= value <= 255 for value in self.movable_prototype_rgb_u8):
            raise ValueError("movable prototype must be a uint8 RGB triple")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    raw = _read_regular_file_once(path)
    if _sha256(raw) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must be one JSON object")
    return value


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    _publish_immutable(path, rfc8785_canonicalize(payload))


def _load_models(config: DDMV14RealizationFidelityConfigV1) -> tuple[Any, Any, dict[str, Any]]:
    upstream = Path(config.upstream_root).resolve()
    modules_path = upstream / "modules.py"
    if not modules_path.is_file():
        raise DirectDescriptionError("frozen scorer modules.py is unavailable")
    sys.path.insert(0, str(upstream))
    try:
        import modules as upstream_modules
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise DirectDescriptionError("frozen scorer runtime imports are unavailable") from exc
    if Path(upstream_modules.__file__).resolve() != modules_path:
        raise DirectDescriptionError("frozen scorer imported non-custodied modules.py")
    torch.set_num_threads(config.scorer_threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    segnet = upstream_modules.SegNet().eval().to("cpu")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    seg_path = Path(upstream_modules.segnet_sd_path).resolve()
    pose_path = Path(upstream_modules.posenet_sd_path).resolve()
    segnet.load_state_dict(load_file(str(seg_path), device="cpu"))
    posenet.load_state_dict(load_file(str(pose_path), device="cpu"))
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    custody = {
        "modules_path": str(modules_path),
        "modules_sha256": _sha256(_read_regular_file_once(modules_path)),
        "segnet_weights_path": str(seg_path),
        "segnet_weights_sha256": _sha256(_read_regular_file_once(seg_path)),
        "posenet_weights_path": str(pose_path),
        "posenet_weights_sha256": _sha256(_read_regular_file_once(pose_path)),
        "device": "cpu",
        "batch_size": config.scorer_batch_size,
        "deterministic_algorithms": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return segnet, posenet, custody


def _forward(segnet: Any, posenet: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    value = np.asarray(camera_pairs)
    if value.dtype != np.uint8 or value.ndim != 5 or value.shape[1:] != (2, 874, 1164, 3):
        raise DirectDescriptionError("v14 scorer requires uint8 [B,2,874,1164,3]")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        cells = segnet(segnet.preprocess_input(tensor)).argmax(dim=1).cpu().numpy().astype(np.uint8)
        pose_output = posenet(posenet.preprocess_input(tensor))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def _measure_candidate(
    *,
    name: str,
    archive: bytes,
    receiver: Any,
    config: DDMV14RealizationFidelityConfigV1,
    root: Path,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    stage = root / "stage_checkpoints" / name
    archive_sha = _sha256(archive)
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if row.get("archive_sha256") != archive_sha or row.get("typed_config_sha256") != config.typed_config_hash():
                raise DirectDescriptionError("v14 batch checkpoint identity differs")
            continue
        local_ids = tuple(range(start, stop))
        source_ids = config.pair_start + np.arange(start, stop, dtype=np.int64)
        camera = receiver.render_camera_pairs(local_ids)
        cells, pose6 = _forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(pose6, replay_pose6):
                raise DirectDescriptionError("v14 deterministic first-batch replay failed")
        target = np.ascontiguousarray(labels[source_ids])
        target_pose = np.ascontiguousarray(poses[source_ids])
        errors = cells != target
        class_rows: dict[str, dict[str, int]] = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = target == class_id
            class_rows[class_name] = {
                "errors": int(np.count_nonzero(errors & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        per_pair = []
        for index, source_id in enumerate(source_ids):
            pair_errors = int(np.count_nonzero(errors[index]))
            pair_pose = float(np.mean(np.square(pose6[index] - target_pose[index])))
            per_pair.append(
                {
                    "local_pair_id": start + index,
                    "source_pair_id": int(source_id),
                    "errors": pair_errors,
                    "sites": int(errors[index].size),
                    "d_seg": f"{pair_errors / errors[index].size:.12f}",
                    "d_pose": f"{pair_pose:.12f}",
                }
            )
        _write_checkpoint(
            checkpoint,
            {
                "schema": "ddm_v14_realization_batch.v1",
                "candidate": name,
                "typed_config_sha256": config.typed_config_hash(),
                "archive_sha256": archive_sha,
                "local_pair_range": [start, stop],
                "errors": int(np.count_nonzero(errors)),
                "sites": int(errors.size),
                "class_rows": class_rows,
                "pose_squared_error_sum": f"{float(np.square(pose6 - target_pose).sum(dtype=np.float64)):.12f}",
                "pose_coordinates": int(pose6.size),
                "cells_sha256": hashlib.sha256(cells.tobytes()).hexdigest(),
                "pose6_sha256": hashlib.sha256(pose6.tobytes()).hexdigest(),
                "per_pair": per_pair,
                "camera_batch_released_after_forward": True,
                "score_claim": False,
            },
        )
    batch_rows = [
        json.loads(_read_regular_file_once(path))
        for path in sorted(stage.glob("batch_*.json"))
    ]
    expected_batches = (config.pair_count + config.scorer_batch_size - 1) // config.scorer_batch_size
    if len(batch_rows) != expected_batches:
        raise DirectDescriptionError("v14 candidate batch coverage is incomplete")
    class_totals = {name: {"errors": 0, "sites": 0} for name in CLASS_NAMES}
    for batch in batch_rows:
        for class_name, row in batch["class_rows"].items():
            class_totals[class_name]["errors"] += int(row["errors"])
            class_totals[class_name]["sites"] += int(row["sites"])
    errors = sum(int(row["errors"]) for row in batch_rows)
    sites = sum(int(row["sites"]) for row in batch_rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in batch_rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batch_rows)
    per_pair = [row for batch in batch_rows for row in batch["per_pair"]]
    return {
        "candidate": name,
        "archive_bytes": len(archive),
        "archive_sha256": archive_sha,
        "d_seg": f"{errors / sites:.12f}",
        "errors": errors,
        "sites": sites,
        "d_pose": f"{pose_sse / pose_coordinates:.12f}",
        "per_stratum": {
            name: {**row, "d_seg": f"{row['errors'] / row['sites']:.12f}"}
            for name, row in class_totals.items()
        },
        "per_pair": per_pair,
        "batch_digest_chain_sha256": hashlib.sha256(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in batch_rows).encode()
        ).hexdigest(),
        "batch_count": len(batch_rows),
        "batch_size": config.scorer_batch_size,
        "all_batches_checkpointed_and_preserved": True,
        "receiver_custody": dict(receiver.custody),
        "byte_streams": recursive_carrier_byte_rows(archive),
        "score_claim": False,
    }


def _historical_rung(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    row = next(value for value in receipt["composition_ladder"] if value["rung"] == name)
    return {
        "candidate": f"v13_{name}_historical_control",
        "archive_bytes": row["archive"]["bytes"],
        "archive_sha256": row["archive"]["sha256"],
        "d_seg": row["bridge"]["segmentation"]["d_seg"],
        "d_pose": row["bridge"]["pose"]["d_pose"],
        "per_stratum": row["bridge"]["segmentation"]["strata"]["target_class"],
        "measurement_path": "legacy scorer-grid receiver; bypasses camera-resolution R-up",
        "comparable_to_v14_full_R": False,
        "score_claim": False,
    }


def _diagnostics(
    *,
    config: DDMV14RealizationFidelityConfigV1,
    v13_receipt: dict[str, Any],
    old_islands: Any,
    fixed_islands: dict[str, Any],
    fixed_both: dict[str, Any],
    labels: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    g1 = np.asarray(old_islands.worldsheet_g1_mask)
    target = np.asarray(labels[config.pair_start : config.pair_start + config.pair_count]) == 3
    target_sites = target.reshape(config.pair_count, -1).sum(axis=1)
    mask_errors = (g1 != target).reshape(config.pair_count, -1).sum(axis=1)
    valid = np.where(target_sites > 0)[0]
    ordered = valid[np.argsort(mask_errors[valid] / target_sites[valid])]
    representative = tuple(int(ordered[index]) for index in np.linspace(0, len(ordered) - 1, 8).astype(int))
    legacy_grid = old_islands.render_pairs(representative)
    legacy_camera = np.empty((len(representative), 2, 874, 1164, 3), dtype=np.uint8)
    for local_index in range(len(representative)):
        for frame_index in range(2):
            legacy_camera[local_index, frame_index] = render_grid_to_camera_uint8(
                legacy_grid[local_index, frame_index]
            )
    legacy_cells, _legacy_pose = _forward(segnet, posenet, legacy_camera)
    target_rep = np.asarray(labels)[config.pair_start + np.asarray(representative)]
    legacy_full_r_errors = (legacy_cells != target_rep).reshape(len(representative), -1).sum(axis=1)
    fixed_by_pair = {int(row["local_pair_id"]): row for row in fixed_islands["per_pair"]}
    historical_islands = next(row for row in v13_receipt["composition_ladder"] if row["rung"] == "islands")
    historical_by_pair = {
        int(row["pair_id"]): row for row in historical_islands["bridge"]["segmentation"]["per_pair"]
    }
    island_rows = []
    for index, pair_id in enumerate(representative):
        sites = int(target_sites[pair_id])
        island_rows.append(
            {
                "local_pair_id": pair_id,
                "source_pair_id": config.pair_start + pair_id,
                "exact_mask_errors": int(mask_errors[pair_id]),
                "exact_mask_error_fraction": f"{mask_errors[pair_id] / (384 * 512):.12f}",
                "target_movable_sites": sites,
                "legacy_scorer_grid_bypass_R_d_seg": historical_by_pair[pair_id]["d_seg"],
                "legacy_flat_paint_full_R_d_seg": f"{legacy_full_r_errors[index] / (384 * 512):.12f}",
                "fixed_camera_paint_full_R_d_seg": fixed_by_pair[pair_id]["d_seg"],
            }
        )
    lane_candidates = sorted(
        {
            int(row.source_pair_id) - config.pair_start
            for row in old_islands.lane_knots
            if config.pair_start <= int(row.source_pair_id) < config.pair_start + config.pair_count
        }
    )
    if not lane_candidates:
        lane_candidates = [max(0, min(config.pair_count - 1, value - config.pair_start)) for value in (448, 472, 496, 511)]
    lane_ids = tuple(
        int(lane_candidates[index]) for index in np.linspace(0, len(lane_candidates) - 1, 4).astype(int)
    )
    both_by_pair = {int(row["local_pair_id"]): row for row in fixed_both["per_pair"]}
    lane_rows = [
        {
            "local_pair_id": pair_id,
            "source_pair_id": config.pair_start + pair_id,
            "fixed_islands_d_seg": fixed_by_pair[pair_id]["d_seg"],
            "fixed_both_d_seg": both_by_pair[pair_id]["d_seg"],
            "delta_d_seg": f"{float(both_by_pair[pair_id]['d_seg']) - float(fixed_by_pair[pair_id]['d_seg']):.12f}",
        }
        for pair_id in lane_ids
    ]
    return {
        "representative_islands": island_rows,
        "lane_windows": lane_rows,
        "stage_chain": [
            "exact G1 binary mask at 384x512",
            "counted three-byte Movable prototype with full amplitude",
            "nearest hard semantic coverage placed at camera 874x1164",
            "uint8 exact prototype survival with no coverage expansion",
            "evaluator-owned bilinear R-down to 384x512",
            "frozen SegNet argmax",
        ],
        "measured_mechanism": "flat prototype/context projection dominates; coverage expansion is not admitted",
    }


def run(config: DDMV14RealizationFidelityConfigV1, root: Path, semantic_argv: list[str]) -> Path:
    storage = _storage_preflight(root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v14_realization_fidelity_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed v14 receipt typed config differs")
        print(json.dumps({"resumed": True, "complete": True, "receipt": str(receipt_path)}))
        return receipt_path
    v13 = _bound_json(Path(config.v13_receipt_path), config.v13_receipt_sha256, "v13 receipt")
    phase = _bound_json(
        Path(config.lane_phase_receipt_path), config.lane_phase_receipt_sha256, "lane phase receipt"
    )
    if (v13.get("typed_config", {}).get("pair_start"), v13.get("typed_config", {}).get("pair_count")) != (
        config.pair_start,
        config.pair_count,
    ):
        raise DirectDescriptionError("v13 receipt window differs from v14 config")
    cache = Path(config.target_cache_path)
    if not cache.is_file() or cache.stat().st_size != config.target_cache_bytes:
        raise DirectDescriptionError("frozen n600 target cache bytes are unavailable")
    if v13.get("target_custody", {}).get("cache_sha256") != config.target_cache_sha256:
        raise DirectDescriptionError("v13 receipt target cache custody differs")
    labels = open_stored_npy_memmap(cache, "lstars")
    poses = open_stored_npy_memmap(cache, "gt_poses")
    archives: dict[str, tuple[bytes, Any]] = {}
    old_receivers: dict[str, Any] = {}
    for name in config.candidate_ladder:
        old_row = next(row for row in v13["composition_ladder"] if row["rung"] == name)
        old_archive = _read_regular_file_once(Path(old_row["archive"]["path"]))
        if _sha256(old_archive) != old_row["archive"]["sha256"]:
            raise DirectDescriptionError(f"v13 {name} archive SHA-256 differs")
        members, _homes = parse_carrier_compose_archive(old_archive)
        old_receiver = receive_carrier_compose_archive(old_archive)
        old_receivers[name] = old_receiver
        colour_by_role = {
            layer.role: tuple(int(value) for value in layer.paint_rgb_u8) for layer in old_receiver.layers
        }
        colour_by_role["Movable"] = tuple(config.movable_prototype_rgb_u8)
        profile = ReceiverRealizationProfileV1(
            role_rgb_u8=tuple(colour_by_role[role] for role in REALIZATION_PAINT_ORDER)
        )
        archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=members[WORLDSHEET_G1_MEMBER],
            lane_programs=_decode_lane_programs(members.get("predict/lane_periodic_programs.ddlp", b"")),
            lane_knots=_decode_lane_knots(members.get("predict/lane_drift_knots.ddlk", b"")),
            realization_profile=profile,
        )
        path = root / f"ddm_v14_{name}_n{config.pair_count}.not_a_candidate.zip.receipt-bytes"
        _publish_immutable(path, archive)
        archives[name] = (archive, receive_carrier_compose_archive(archive))
    _write_checkpoint(
        root / "stage_checkpoints" / "00_receiver_closed_archives.json",
        {
            "schema": "ddm_v14_receiver_closed_archives.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "archives": {
                name: {
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                    "receiver_custody": dict(receiver.custody),
                }
                for name, (archive, receiver) in archives.items()
            },
        },
    )
    measured_paths = {
        name: root / "stage_checkpoints" / f"01_{name}_measurement.json" for name in config.candidate_ladder
    }
    missing = [name for name, path in measured_paths.items() if not path.exists()]
    if missing:
        name = missing[0]
        segnet, posenet, scorer_custody = _load_models(config)
        result = _measure_candidate(
            name=name,
            archive=archives[name][0],
            receiver=archives[name][1],
            config=config,
            root=root,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
        )
        result["scorer_custody"] = scorer_custody
        _write_checkpoint(measured_paths[name], result)
        print(json.dumps({"resumed": False, "complete": False, "measured_stage": name}))
        return measured_paths[name]
    measured = {name: json.loads(_read_regular_file_once(path)) for name, path in measured_paths.items()}
    segnet, posenet, scorer_custody = _load_models(config)
    diagnostics = _diagnostics(
        config=config,
        v13_receipt=v13,
        old_islands=old_receivers["islands"],
        fixed_islands=measured["islands"],
        fixed_both=measured["both"],
        labels=labels,
        segnet=segnet,
        posenet=posenet,
    )
    g4_runtime_path = Path(config.upstream_root).resolve().parent / G4_RECEIPT_PATH
    g4 = _bound_json(g4_runtime_path, G4_RECEIPT_SHA256, "landed G4 stationarity receipt")
    if g4.get("score_claim") is not False or g4.get("pointer") != f"{POINTER_SCORE_TEXT} [contest-CPU]":
        raise DirectDescriptionError("landed G4 stationarity receipt authority differs")
    selected = min(measured.values(), key=lambda row: float(row["d_seg"]))
    gate_pass = float(selected["d_seg"]) <= config.dseg_gate and int(selected["archive_bytes"]) <= config.archive_box_bytes
    producer_paths = (
        REPO_ROOT / "tools/measure_ddm_v14_realization_fidelity.py",
        REPO_ROOT / "src/tac/optimization/direct_description_carrier_compose.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": "ddm_v14_realization_fidelity",
        "tasks": [603, 613, 578],
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "producer_custody": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(path)),
            }
            for path in producer_paths
        ],
        "historical_controls": [_historical_rung(v13, name) for name in ("base", "islands", "lane", "both")],
        "fixed_ladder": [measured[name] for name in config.candidate_ladder],
        "selected_candidate": selected["candidate"],
        "diagnostics": diagnostics,
        "fork": {
            "condition": "full-n600 selected fixed composed d_seg <=0.00116 at <=200000 exact bytes",
            "passed": gate_pass,
            "disposition": (
                "FLAG_MAIN_FOR_R6_EXACT_EVAL_NO_MODAL_DISPATCH"
                if gate_pass
                else "FORMULATION_SCOPED_IRREDUCIBLE_PROJECTION_LOSS_DIRECT_RGB_SCORER_SOLVE_OPEN"
            ),
        },
        "lane_successor": {
            "requested": "bev_curvature_dash_comb_range_gate_anisotropic_ar1_whitened_innovations_road_polytope",
            "g4_dependency_landed": True,
            "measured_in_this_receipt": False,
            "reason": (
                "G4 stationarity landed after the base stages; its three priced static fields are measured "
                "through this repaired receiver in a separately checkpointed companion receipt"
            ),
            "g4_receipt": {"path": G4_RECEIPT_PATH, "sha256": G4_RECEIPT_SHA256},
            "companion_receipt_expected": (
                ".omx/research/ddm_v14_g4_receiver_projection_n600_20260722T221500Z/"
                "ddm_v14_g4_receiver_projection_receipt.json"
            ),
            "receiver_fixed_pre_addendum_both_measured": True,
            "raw_phase_control_receipt": {
                "path": config.lane_phase_receipt_path,
                "sha256": config.lane_phase_receipt_sha256,
                "verdict": phase.get("verdict"),
                "verdict_scope": phase.get("verdict_scope"),
            },
        },
        "fail_closed_mutation_proof": prove_carrier_archive_fail_closed(archives[selected["candidate"]][0]),
        "scorer_custody": scorer_custody,
        "target_custody": {
            "path": str(cache),
            "bytes": config.target_cache_bytes,
            "sha256": config.target_cache_sha256,
            "mutated": False,
        },
        "storage_preflight": storage,
        "resume": {
            "batch_size": config.scorer_batch_size,
            "per_batch_checkpoints": True,
            "all_preserved": True,
            "candidate_stage_limit_per_invocation": config.max_candidate_stages_per_invocation,
        },
        "blocker_delta": "V13 scorer-grid bypass and inherited flat Movable paint are replaced by a counted camera-res profile; residual is measured receiver projection, with direct RGB scorer solve open.",
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md",
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",
            config.v13_receipt_path,
            config.lane_phase_receipt_path,
            config.target_cache_path,
            G4_RECEIPT_PATH,
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_immutable(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "complete": True, "receipt": str(receipt_path)}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    raw = _read_regular_file_once(args.config)
    config = DDMV14RealizationFidelityConfigV1.model_validate_json(raw)
    semantic_argv = [
        "tools/measure_ddm_v14_realization_fidelity.py",
        "--config",
        str(args.config),
        "--output-directory",
        str(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
