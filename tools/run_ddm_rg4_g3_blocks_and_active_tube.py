#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the RG3 obstruction harvest and two-candidate PC1/Pose6 tube measure."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_ms2r_r3_366box_typed_fisher_g4_waterfill import (  # noqa: E402
    _validate_sources as validate_ms2r_r3_sources,
)
from tac.optimization.ddm_pc1_pose_stream import (  # noqa: E402
    DDMPC1TrainableParameterMapV1,
    PC1PosePacketV1,
    serialize_pc1_packet,
)
from tac.optimization.ddm_pc2_pose_descent import (  # noqa: E402
    bit_reversal_knot_order,
    four_pair_batch_for_knot,
    score_domain_action,
)
from tac.optimization.ddm_rg4_g3_blocks_and_active_tube import (  # noqa: E402
    DDMRG4Error,
    active_tube_report,
    build_source_local_composition_archive,
    canonical_bytes,
    parse_source_local_composition_archive,
    receive_source_local_pc1_camera_pairs,
    rg3_typed_exclusions,
    sha256_bytes,
)
from tac.optimization.ddm_ws1_warm_start import receive_ws1_warm_start_archive  # noqa: E402
from tools.measure_ddm_menu1_realized_flip_menu import _forward  # noqa: E402
from tools.summarize_ddm_ms6_receiver_support import build_summary  # noqa: E402

SCHEMA = "ddm_rg4_g3_blocks_and_active_tube_config.v1"
RECEIPT_SCHEMA = "ddm_rg4_pc1_pose6_active_tube_receipt.v1"
VERDICT_SCHEMA = "ddm_rg4_pc1_pose6_n600_verdict.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
MIN_AVAILABLE_BYTES = 20 * 1024**3
MIN_STORAGE_BYTES = 20 * 1024**3
POSE_AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ONE_QUANTUM_Q = 256


class RunnerError(DDMRG4Error):
    """Raised when typed config, custody, or resumable state differs."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RunnerError(f"immutable output differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _atomic_bytes(path, canonical_bytes(dict(value)))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"JSON object unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"JSON artifact must be one object: {path}")
    return value


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO / value


def _validate_binding(row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    if set(row) != {"path", "bytes", "sha256", "schema", "role"}:
        raise RunnerError("artifact binding schema differs")
    path = _resolve(str(row["path"]))
    if not path.is_file():
        raise RunnerError(f"bound artifact is missing: {path}")
    observed_bytes, observed_sha = _sha256_file(path)
    if observed_bytes != row["bytes"] or observed_sha != row["sha256"]:
        raise RunnerError(f"bound artifact drifted: {path}")
    return path, {
        "path": str(row["path"]),
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "schema": row["schema"],
        "role": row["role"],
    }


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = path.read_bytes()
    config = _read_json(path)
    required = {
        "schema",
        "run_id",
        "lane_id",
        "output_root",
        "research_root",
        "authority",
        "artifacts",
        "checkpoint_roots",
        "pair_count",
        "verdict_batch",
        "local_batch",
        "knot_count",
        "one_quantum_q",
        "maximum_accepted_steps",
        "maximum_candidate_evaluations",
        "seed",
        "torch_threads",
        "own_python",
        "upstream_root",
        "research_only",
        "paid_dispatch",
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
        "main_landing_review_required",
    }
    if set(config) != required or config.get("schema") != SCHEMA:
        raise RunnerError("typed RG4 config schema/inventory differs")
    expected = {
        "pair_count": 600,
        "verdict_batch": 32,
        "local_batch": 4,
        "knot_count": 32,
        "one_quantum_q": ONE_QUANTUM_Q,
        "maximum_accepted_steps": 8,
        "maximum_candidate_evaluations": 384,
        "seed": 0,
        "torch_threads": 4,
        "research_only": True,
        "paid_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RunnerError(f"typed RG4 config field differs: {key}")
    expected_artifacts = {
        "authority",
        "g3_registry",
        "ms2r_config",
        "pc1_admission",
        "pose_metric",
        "rg2_assignment",
        "rg3_assignment",
        "rg3_summary",
        "rg3_table",
        "target_cache",
        "menu1_config",
        "inner_jacobian_status",
        "w_joint",
        "w_seg",
    }
    if set(config["artifacts"]) != expected_artifacts:
        raise RunnerError("typed RG4 artifact inventory differs")
    bindings = {}
    for name, row in sorted(config["artifacts"].items()):
        _path, bindings[name] = _validate_binding(row)
    authority_path, _authority = _validate_binding(config["authority"])
    if authority_path != _resolve(config["artifacts"]["authority"]["path"]):
        raise RunnerError("delegated authority aliases differ")
    if Path(sys.executable).absolute() != Path(config["own_python"]).absolute():
        raise RunnerError(f"RG4 must run in owned venv {config['own_python']}; got {sys.executable}")
    custody = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "schema": SCHEMA,
        "bindings": bindings,
    }
    return config, custody


def _memory_receipt(stage: str) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RunnerError("RG4 scorer preflight requires psutil") from exc
    memory = psutil.virtual_memory()
    available = int(memory.available)
    if available < MIN_AVAILABLE_BYTES:
        raise RunnerError(f"REFUSE_RG4_MEMORY_{stage}: {available} < {MIN_AVAILABLE_BYTES}")
    return {
        "stage": stage,
        "total_bytes": int(memory.total),
        "available_bytes": available,
        "required_available_bytes": MIN_AVAILABLE_BYTES,
        "admission": True,
        "source": "psutil.virtual_memory",
    }


def _configure_torch(config: Mapping[str, Any]) -> int:
    import torch

    torch.set_num_threads(int(config["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    observed = int(torch.get_num_threads())
    if observed != config["torch_threads"]:
        raise RunnerError("RG4 torch thread count differs")
    return observed


def _load_scorers(upstream_root: str) -> tuple[Any, Any, dict[str, Any]]:
    from safetensors.torch import load_file

    upstream = Path(upstream_root)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules as upstream_modules

    if Path(upstream_modules.__file__).resolve() != (upstream / "modules.py").resolve():
        raise RunnerError("frozen scorer imported noncustodied modules.py")
    segnet = upstream_modules.SegNet().eval().to("cpu")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    segnet.load_state_dict(load_file(str(upstream_modules.segnet_sd_path), device="cpu"))
    posenet.load_state_dict(load_file(str(upstream_modules.posenet_sd_path), device="cpu"))
    for scorer in (segnet, posenet):
        for parameter in scorer.parameters():
            parameter.requires_grad = False
    custody = {}
    for name, scorer_path in {
        "modules": upstream / "modules.py",
        "segnet": upstream / "models" / "segnet.safetensors",
        "posenet": upstream / "models" / "posenet.safetensors",
    }.items():
        size, digest = _sha256_file(scorer_path)
        custody[name] = {"path": str(scorer_path), "bytes": size, "sha256": digest}
    return segnet, posenet, custody


def _parameter_map(config: Mapping[str, Any]) -> DDMPC1TrainableParameterMapV1:
    pc1 = _read_json(_resolve(config["artifacts"]["pc1_admission"]["path"]))
    row = pc1["parameter_map"]
    if (
        pc1.get("schema") != "ddm_pc1_pose_stream_admission.v1"
        or row.get("descent_trainable") is not True
        or row.get("coordinate_count") != 320
    ):
        raise RunnerError("PC1 trainable parameter-map custody differs")
    return DDMPC1TrainableParameterMapV1(
        pair_count=600,
        knot_count=32,
        xi_scales=tuple(float(value) for value in row["xi_scales"]),
        residual_scale=float(row["residual_scale"]),
    )


def _packet(parameter_map: DDMPC1TrainableParameterMapV1, q_xi: np.ndarray) -> PC1PosePacketV1:
    q = np.asarray(q_xi)
    if q.shape != (32, 6):
        raise RunnerError("RG4 q_xi geometry differs")
    return PC1PosePacketV1(
        active=True,
        pair_count=600,
        xi_scales=parameter_map.xi_scales,
        residual_scale=parameter_map.residual_scale,
        q_xi=np.ascontiguousarray(q, dtype=np.int16),
        q_luma_phase=np.zeros((32, 4), dtype=np.int8),
    )


def _movable_layer(receiver: Any) -> Any:
    try:
        return next(layer for layer in receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise RunnerError("candidate has no Movable receiver layer") from exc


def _parent_batch(
    receiver: Any,
    movable_layer: Any,
    pair_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    ids = tuple(int(value) for value in pair_ids)
    parent = receiver.render_camera_pairs(ids)
    masks = np.stack(
        [
            receiver._mask_for_layer(
                movable_layer,
                pair_id,
                replace_g1_movable=True,
            )
            for pair_id in ids
        ],
        axis=0,
    ).astype(np.bool_)
    return parent, masks


def _zero_home(
    *,
    parent: np.ndarray,
    masks: np.ndarray,
    pair_ids: Sequence[int],
    parameter_map: DDMPC1TrainableParameterMapV1,
) -> np.ndarray:
    from tac.optimization.ddm_pc1_pose_stream import receive_pc1_camera_pairs

    return receive_pc1_camera_pairs(
        parent_camera=parent,
        packet=_packet(parameter_map, np.zeros((32, 6), dtype=np.int16)),
        pair_ids=pair_ids,
        movable_masks=masks,
    )


def _render(
    *,
    parent: np.ndarray,
    masks: np.ndarray,
    zero_home: np.ndarray,
    pair_ids: Sequence[int],
    packet: PC1PosePacketV1,
) -> np.ndarray:
    return receive_source_local_pc1_camera_pairs(
        parent_camera=parent,
        packet=packet,
        pair_ids=pair_ids,
        movable_masks=masks,
        absolute_zero_home=zero_home,
    )


def _composition(
    *,
    parent_bytes: bytes,
    parent_sha256: str,
    packet: PC1PosePacketV1,
) -> bytes:
    archive = build_source_local_composition_archive(
        parent_archive=parent_bytes,
        parent_sha256=parent_sha256,
        packet=packet,
    )
    parsed_parent, parsed_packet, _manifest = parse_source_local_composition_archive(archive)
    if parsed_parent != parent_bytes or serialize_pc1_packet(parsed_packet) != serialize_pc1_packet(packet):
        raise RunnerError("source-local composition parse-back differs")
    return archive


def _score_batch(
    *,
    camera: np.ndarray,
    pair_ids: Sequence[int],
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
    archive_bytes: int,
    stage: str,
) -> dict[str, Any]:
    ids = tuple(int(value) for value in pair_ids)
    camera_array = np.asarray(camera)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or camera_array.dtype != np.uint8
        or camera_array.ndim != 5
        or camera_array.shape[0] != len(ids)
        or camera_array.shape[-1] != 3
        or archive_bytes <= 0
    ):
        raise RunnerError(f"RG4 scorer input geometry/custody differs: {stage}")
    memory = _memory_receipt(stage)
    cells, pose6 = _forward(scorers[0], scorers[1], camera_array)
    cells = np.asarray(cells)
    pose6 = np.asarray(pose6, dtype=np.float64)
    if cells.shape != (len(ids), 384, 512) or pose6.shape != (len(ids), 6) or not np.all(np.isfinite(pose6)):
        raise RunnerError(f"RG4 scorer output geometry/finiteness differs: {stage}")
    target = np.asarray(labels[list(ids)], dtype=np.uint8)
    target_pose = np.asarray(poses[list(ids)], dtype=np.float64)
    if target.shape != cells.shape or target_pose.shape != pose6.shape:
        raise RunnerError(f"RG4 scorer target geometry differs: {stage}")
    errors = int(np.count_nonzero(cells != target))
    sites = int(cells.size)
    pose_squared = np.square(pose6 - target_pose)
    pose_sse_by_dimension = pose_squared.sum(axis=0, dtype=np.float64)
    pose_sse = float(pose_sse_by_dimension.sum())
    d_seg = errors / sites
    d_pose = pose_sse / int(pose6.size)
    return {
        "pair_ids": list(ids),
        "errors": errors,
        "sites": sites,
        "pose_squared_error_sum": pose_sse,
        "pose_squared_error_sum_by_dimension": pose_sse_by_dimension.tolist(),
        "pose_coordinates": int(pose6.size),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "advisory_action": score_domain_action(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
        ),
        "camera_sha256": sha256_bytes(camera_array.tobytes(order="C")),
        "pose6": pose6,
        "memory_preflight": memory,
    }


def _checkpoint_path(root: Path, cursor: int) -> Path:
    return root / "checkpoints" / f"schedule_{cursor:03d}.json"


def _resume_search(
    *,
    root: Path,
    config_sha256: str,
    base_sha256: str,
    parameter_map: DDMPC1TrainableParameterMapV1,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    latest: dict[str, Any] | None = None
    gap = False
    for cursor in range(33):
        path = _checkpoint_path(root, cursor)
        if not path.exists():
            gap = True
            continue
        if gap:
            raise RunnerError("RG4 schedule checkpoints contain a gap")
        row = _read_json(path)
        if (
            row.get("schema") != "ddm_rg4_pc1_local_descent_checkpoint.v1"
            or row.get("typed_config_sha256") != config_sha256
            or row.get("base_archive_sha256") != base_sha256
            or row.get("schedule_cursor") != cursor
            or row.get("status") != "PRESERVED_COMPLETE_STAGE_STATE"
        ):
            raise RunnerError(f"RG4 checkpoint identity differs: {path}")
        packet = _packet(parameter_map, np.asarray(row["q_xi"], dtype=np.int16))
        if sha256_bytes(serialize_pc1_packet(packet)) != row.get("packet_sha256"):
            raise RunnerError(f"RG4 checkpoint packet differs: {path}")
        latest = row
    if latest is None:
        q_xi = np.zeros((32, 6), dtype=np.int16)
        payload = {
            "schema": "ddm_rg4_pc1_local_descent_checkpoint.v1",
            "typed_config_sha256": config_sha256,
            "base_archive_sha256": base_sha256,
            "schedule_cursor": 0,
            "candidate_evaluations": 0,
            "accepted_steps": [],
            "stage_proposals": [],
            "q_xi": q_xi.tolist(),
            "packet_sha256": sha256_bytes(serialize_pc1_packet(_packet(parameter_map, q_xi))),
            "status": "PRESERVED_COMPLETE_STAGE_STATE",
            "all_prior_stage_checkpoints_preserved": True,
            "score_claim": False,
        }
        _write_json(_checkpoint_path(root, 0), payload)
        return q_xi, [], 0
    return (
        np.asarray(latest["q_xi"], dtype=np.int16),
        [dict(value) for value in latest["accepted_steps"]],
        int(latest["schedule_cursor"]),
    )


def _save_search_checkpoint(
    *,
    root: Path,
    config_sha256: str,
    base_sha256: str,
    parameter_map: DDMPC1TrainableParameterMapV1,
    q_xi: np.ndarray,
    accepted_steps: Sequence[Mapping[str, Any]],
    cursor: int,
    stage_proposals: Sequence[Mapping[str, Any]],
) -> None:
    packet = _packet(parameter_map, q_xi)
    payload = {
        "schema": "ddm_rg4_pc1_local_descent_checkpoint.v1",
        "typed_config_sha256": config_sha256,
        "base_archive_sha256": base_sha256,
        "schedule_cursor": cursor,
        "candidate_evaluations": cursor * 12,
        "accepted_steps": list(accepted_steps),
        "stage_proposals": list(stage_proposals),
        "q_xi": np.asarray(q_xi, dtype=np.int16).tolist(),
        "packet_sha256": sha256_bytes(serialize_pc1_packet(packet)),
        "status": "PRESERVED_COMPLETE_STAGE_STATE",
        "all_prior_stage_checkpoints_preserved": True,
        "score_claim": False,
    }
    _write_json(_checkpoint_path(root, cursor), payload)


def _proposal_row(
    *,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    coordinate_id: str,
    knot_id: int,
    axis_id: int,
    direction: int,
    candidate_archive_sha256: str,
    base_sha256: str,
    base_bytes: int,
) -> dict[str, Any]:
    return {
        "coordinate_id": coordinate_id,
        "knot_id": knot_id,
        "pose_axis": POSE_AXES[axis_id],
        "pose_output_dimension": axis_id,
        "direction": direction,
        "q_increment": direction * ONE_QUANTUM_Q,
        "physical_increment": "ONE_QUANTUM",
        "pair_ids": candidate["pair_ids"],
        "receiver_visible": candidate["camera_sha256"] != current["camera_sha256"],
        "d_seg": candidate["d_seg"],
        "d_pose": candidate["d_pose"],
        "archive_bytes": candidate["archive_bytes"],
        "archive_sha256": candidate_archive_sha256,
        "base_archive_sha256": base_sha256,
        "base_archive_bytes": base_bytes,
        "seg_delta": float(candidate["d_seg"]) - float(current["d_seg"]),
        "pose_delta": float(candidate["d_pose"]) - float(current["d_pose"]),
        "pose_delta_by_dimension": [
            (
                float(candidate["pose_squared_error_sum_by_dimension"][index])
                - float(current["pose_squared_error_sum_by_dimension"][index])
            )
            / len(candidate["pair_ids"])
            for index in range(6)
        ],
        "rate_bytes_delta_from_current": int(candidate["archive_bytes"]) - int(current["archive_bytes"]),
        "rate_bytes_delta_from_base": int(candidate["archive_bytes"]) - base_bytes,
        "joint_delta": float(candidate["advisory_action"]) - float(current["advisory_action"]),
        "candidate_local_descent_rate": (
            (float(candidate["advisory_action"]) - float(current["advisory_action"]))
            / max(abs(int(candidate["archive_bytes"]) - int(current["archive_bytes"])), 1)
        ),
        "camera_sha256": candidate["camera_sha256"],
        "memory_preflight": candidate["memory_preflight"],
        "selection_surface": "four-pair exact receiver/frozen-scorer local batch",
        "score_claim": False,
    }


def _select(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    admissible = [
        row
        for row in rows
        if row["receiver_visible"] and float(row["pose_delta"]) < -1.0e-12 and float(row["joint_delta"]) < -1.0e-12
    ]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda row: (
            float(row["joint_delta"]),
            float(row["pose_delta"]),
            float(row["seg_delta"]),
            str(row["coordinate_id"]),
            int(row["direction"]),
        ),
    )


def _run_search(
    *,
    candidate_id: str,
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    receiver: Any,
    movable_layer: Any,
    parent_bytes: bytes,
    base_sha256: str,
    parameter_map: DDMPC1TrainableParameterMapV1,
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    q_xi, accepted, cursor = _resume_search(
        root=root,
        config_sha256=config_sha256,
        base_sha256=base_sha256,
        parameter_map=parameter_map,
    )
    order = bit_reversal_knot_order(32)
    while cursor < 32:
        knot_id = order[cursor]
        pair_ids = four_pair_batch_for_knot(knot_id)
        parent, masks = _parent_batch(receiver, movable_layer, pair_ids)
        zero_home = _zero_home(
            parent=parent,
            masks=masks,
            pair_ids=pair_ids,
            parameter_map=parameter_map,
        )
        current_packet = _packet(parameter_map, q_xi)
        current_archive = _composition(
            parent_bytes=parent_bytes,
            parent_sha256=base_sha256,
            packet=current_packet,
        )
        current_camera = _render(
            parent=parent,
            masks=masks,
            zero_home=zero_home,
            pair_ids=pair_ids,
            packet=current_packet,
        )
        current = _score_batch(
            camera=current_camera,
            pair_ids=pair_ids,
            labels=labels,
            poses=poses,
            scorers=scorers,
            archive_bytes=len(current_archive),
            stage=f"{candidate_id}_SCHEDULE_{cursor:02d}_CURRENT",
        )
        proposal_rows = []
        proposal_q: dict[str, np.ndarray] = {}
        for axis_id, axis in enumerate(POSE_AXES):
            for direction in (-1, 1):
                candidate_q = q_xi.copy()
                candidate_q[knot_id, axis_id] += direction * ONE_QUANTUM_Q
                packet = _packet(parameter_map, candidate_q)
                archive = _composition(
                    parent_bytes=parent_bytes,
                    parent_sha256=base_sha256,
                    packet=packet,
                )
                camera = _render(
                    parent=parent,
                    masks=masks,
                    zero_home=zero_home,
                    pair_ids=pair_ids,
                    packet=packet,
                )
                measured = _score_batch(
                    camera=camera,
                    pair_ids=pair_ids,
                    labels=labels,
                    poses=poses,
                    scorers=scorers,
                    archive_bytes=len(archive),
                    stage=f"{candidate_id}_SCHEDULE_{cursor:02d}_{axis}_{direction:+d}",
                )
                coordinate_id = f"ddm.pc1.knot.{knot_id:03d}.xi.{axis}"
                row = _proposal_row(
                    current=current,
                    candidate=measured,
                    coordinate_id=coordinate_id,
                    knot_id=knot_id,
                    axis_id=axis_id,
                    direction=direction,
                    candidate_archive_sha256=sha256_bytes(archive),
                    base_sha256=base_sha256,
                    base_bytes=len(parent_bytes),
                )
                proposal_rows.append(row)
                proposal_q[f"{coordinate_id}:{direction}"] = candidate_q
                del camera, measured
                gc.collect()
        winner = _select(proposal_rows) if len(accepted) < 8 else None
        if winner is not None:
            key = f"{winner['coordinate_id']}:{winner['direction']}"
            q_xi = proposal_q[key]
            accepted.append(
                {
                    **dict(winner),
                    "accepted_step": len(accepted) + 1,
                    "schedule_cursor": cursor,
                }
            )
        cursor += 1
        _save_search_checkpoint(
            root=root,
            config_sha256=config_sha256,
            base_sha256=base_sha256,
            parameter_map=parameter_map,
            q_xi=q_xi,
            accepted_steps=accepted,
            cursor=cursor,
            stage_proposals=proposal_rows,
        )
        print(
            json.dumps(
                {
                    "stage": "local_descent",
                    "candidate_id": candidate_id,
                    "schedule_cursor": cursor,
                    "candidate_evaluations": cursor * 12,
                    "accepted_steps": len(accepted),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del parent, masks, zero_home, current_camera, current
        gc.collect()
    return q_xi, accepted


def _exact_chunk_path(root: Path, label: str, start: int, stop: int) -> Path:
    return root / "verdicts" / label / f"chunk_{start:04d}_{stop:04d}.json"


def _exact_verdict(
    *,
    candidate_id: str,
    label: str,
    root: Path,
    receiver: Any,
    movable_layer: Any,
    parent_bytes: bytes,
    base_sha256: str,
    parameter_map: DDMPC1TrainableParameterMapV1,
    q_xi: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
    centers: np.ndarray,
    factors: np.ndarray,
    tube_radius: float,
    scorers: tuple[Any, Any],
) -> dict[str, Any]:
    verdict_path = root / "verdicts" / label / "n600.json"
    packet = _packet(parameter_map, q_xi)
    is_source = label == "source"
    if is_source:
        archive = parent_bytes
    else:
        archive = _composition(
            parent_bytes=parent_bytes,
            parent_sha256=base_sha256,
            packet=packet,
        )
    archive_sha256 = sha256_bytes(archive)
    if verdict_path.exists():
        value = _read_json(verdict_path)
        if (
            value.get("schema") != VERDICT_SCHEMA
            or value.get("archive_sha256") != archive_sha256
            or value.get("candidate_id") != candidate_id
            or value.get("label") != label
        ):
            raise RunnerError("preserved RG4 n600 verdict differs")
        return value
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        path = _exact_chunk_path(root, label, start, stop)
        if path.exists():
            row = _read_json(path)
            if row.get("archive_sha256") != archive_sha256 or row.get("pair_range") != [start, stop]:
                raise RunnerError(f"preserved RG4 exact chunk differs: {path}")
            rows.append(row)
            continue
        ids = tuple(range(start, stop))
        parent, masks = _parent_batch(receiver, movable_layer, ids)
        if is_source:
            camera = parent
        else:
            zero_home = _zero_home(
                parent=parent,
                masks=masks,
                pair_ids=ids,
                parameter_map=parameter_map,
            )
            camera = _render(
                parent=parent,
                masks=masks,
                zero_home=zero_home,
                pair_ids=ids,
                packet=packet,
            )
        measured = _score_batch(
            camera=camera,
            pair_ids=ids,
            labels=labels,
            poses=poses,
            scorers=scorers,
            archive_bytes=len(archive),
            stage=f"{candidate_id}_{label}_N600_{start:04d}_{stop:04d}",
        )
        pose6 = measured.pop("pose6")
        tube = active_tube_report(
            pose6=pose6,
            centers=np.asarray(centers[start:stop]),
            low_rank_factors=np.asarray(factors[start:stop]),
            tube_radius=tube_radius,
        )
        row = {
            "schema": "ddm_rg4_pc1_pose6_n600_chunk.v1",
            "candidate_id": candidate_id,
            "label": label,
            "pair_range": [start, stop],
            "archive_sha256": archive_sha256,
            "errors": measured["errors"],
            "sites": measured["sites"],
            "pose_squared_error_sum": measured["pose_squared_error_sum"],
            "pose_squared_error_sum_by_dimension": measured["pose_squared_error_sum_by_dimension"],
            "pose_coordinates": measured["pose_coordinates"],
            "camera_sha256": measured["camera_sha256"],
            "memory_preflight": measured["memory_preflight"],
            "tube": tube,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        _write_json(path, row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "n600_chunk",
                    "candidate_id": candidate_id,
                    "label": label,
                    "pair_range": [start, stop],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del parent, masks, camera, measured, pose6
        gc.collect()
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse_by_dimension = np.sum(
        np.asarray(
            [row["pose_squared_error_sum_by_dimension"] for row in rows],
            dtype=np.float64,
        ),
        axis=0,
    )
    pose_sse = float(pose_sse_by_dimension.sum())
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    projected_sums = np.sum(
        np.asarray(
            [
                [
                    dim["mean_projected_squared_contribution"] * row["tube"]["pair_count"]
                    for dim in row["tube"]["dimensions"]
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    radius_squared = tube_radius**2
    equal_share = radius_squared / 6.0
    dimension_rows = [
        {
            "pose_output_dimension": dimension,
            "pose_axis": POSE_AXES[dimension],
            "pose_squared_error_sum": float(pose_sse_by_dimension[dimension]),
            "d_pose_dimension_mse": float(pose_sse_by_dimension[dimension] / 600),
            "d_pose_global_contribution": float(pose_sse_by_dimension[dimension] / 3600),
            "mean_projected_squared_contribution": float(projected_sums[dimension] / 600),
            "equal_rank_share_budget": equal_share,
            "equal_rank_share_slack": float(equal_share - projected_sums[dimension] / 600),
            "active_under_equal_rank_share_diagnostic": bool(projected_sums[dimension] / 600 >= equal_share),
        }
        for dimension in range(6)
    ]
    tube = {
        "metric": "sum_i (L_i @ (pose6_i-center_i))^2",
        "pair_count": 600,
        "tube_radius": tube_radius,
        "tube_radius_squared": radius_squared,
        "overall_membership_rule": "ALL_PAIR_QUADRATICS_LE_RADIUS_SQUARED",
        "all_pairs_inside": all(row["tube"]["all_pairs_inside"] for row in rows),
        "inside_pair_count": sum(int(row["tube"]["inside_pair_count"]) for row in rows),
        "outside_pair_count": sum(int(row["tube"]["outside_pair_count"]) for row in rows),
        "inside_pair_fraction": sum(int(row["tube"]["inside_pair_count"]) for row in rows) / 600,
        "mean_pair_quadratic": float(projected_sums.sum() / 600),
        "max_pair_quadratic": max(float(row["tube"]["max_pair_quadratic"]) for row in rows),
        "minimum_full_quadratic_slack": min(float(row["tube"]["minimum_full_quadratic_slack"]) for row in rows),
        "dimension_activity_rule": (
            "DIAGNOSTIC_ONLY: equal share of radius^2 across six factor-output "
            "dimensions; does not replace the full quadratic membership rule"
        ),
        "active_dimension_count": sum(bool(row["active_under_equal_rank_share_diagnostic"]) for row in dimension_rows),
        "dimensions": dimension_rows,
    }
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    verdict = {
        "schema": VERDICT_SCHEMA,
        "candidate_id": candidate_id,
        "label": label,
        "num_pairs": 600,
        "batch_size": 32,
        "maximum_rgb_chunks_resident": 1,
        "base_archive_sha256": base_sha256,
        "archive_sha256": archive_sha256,
        "archive_bytes": len(archive),
        "packet_sha256": sha256_bytes(serialize_pc1_packet(packet)),
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "pose_squared_error_sum": pose_sse,
        "pose_squared_error_sum_by_dimension": pose_sse_by_dimension.tolist(),
        "pose_coordinates": pose_coordinates,
        "d_pose": d_pose,
        "advisory_action": score_domain_action(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=len(archive),
        ),
        "active_tube": tube,
        "chunk_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    _write_json(verdict_path, verdict)
    if not is_source:
        _atomic_bytes(
            root / "archives" / "final_pc1_source_local.zip.receipt-bytes",
            archive,
        )
    return verdict


def _rate_rows(
    source: Mapping[str, Any],
    final: Mapping[str, Any],
) -> list[dict[str, Any]]:
    delta_bytes = int(final["archive_bytes"]) - int(source["archive_bytes"])
    rows = [
        {
            "scope": "JOINT_SCORE_DOMAIN",
            "pose_output_dimension": None,
            "base_archive_sha256": source["base_archive_sha256"],
            "base_archive_bytes": source["archive_bytes"],
            "candidate_archive_sha256": final["archive_sha256"],
            "candidate_archive_bytes": final["archive_bytes"],
            "delta_bytes": delta_bytes,
            "delta_S": float(final["advisory_action"]) - float(source["advisory_action"]),
            "delta_S_per_added_byte": (
                (float(final["advisory_action"]) - float(source["advisory_action"])) / delta_bytes
                if delta_bytes
                else None
            ),
        }
    ]
    for dimension in range(6):
        source_dim = source["active_tube"]["dimensions"][dimension]
        final_dim = final["active_tube"]["dimensions"][dimension]
        delta_contribution = float(final_dim["d_pose_global_contribution"]) - float(
            source_dim["d_pose_global_contribution"]
        )
        rows.append(
            {
                "scope": "POSE_DPOSE_CONTRIBUTION",
                "pose_output_dimension": dimension,
                "pose_axis": POSE_AXES[dimension],
                "base_archive_sha256": source["base_archive_sha256"],
                "base_archive_bytes": source["archive_bytes"],
                "candidate_archive_sha256": final["archive_sha256"],
                "candidate_archive_bytes": final["archive_bytes"],
                "delta_bytes": delta_bytes,
                "delta_d_pose_global_contribution": delta_contribution,
                "delta_d_pose_global_contribution_per_added_byte": (
                    delta_contribution / delta_bytes if delta_bytes else None
                ),
                "final_equal_rank_share_slack": final_dim["equal_rank_share_slack"],
                "final_active_under_equal_rank_share_diagnostic": final_dim["active_under_equal_rank_share_diagnostic"],
            }
        )
    return rows


def _run_candidate(
    *,
    candidate_id: str,
    binding_name: str,
    config: Mapping[str, Any],
    config_custody: Mapping[str, Any],
    output_root: Path,
    labels: np.ndarray,
    poses: np.ndarray,
    centers: np.ndarray,
    factors: np.ndarray,
    tube_radius: float,
    parameter_map: DDMPC1TrainableParameterMapV1,
    scorers: tuple[Any, Any],
) -> dict[str, Any]:
    root = output_root / "candidates" / candidate_id
    binding = config["artifacts"][binding_name]
    parent_bytes = _resolve(binding["path"]).read_bytes()
    base_sha256 = sha256_bytes(parent_bytes)
    receiver = receive_ws1_warm_start_archive(parent_bytes)
    if receiver.archive != parent_bytes or receiver.parsed.exact_reemit() != parent_bytes:
        raise RunnerError(f"{candidate_id} WS1 receiver parse/re-emit differs")
    movable_layer = _movable_layer(receiver)
    q_xi, accepted = _run_search(
        candidate_id=candidate_id,
        root=root,
        config=config,
        config_sha256=str(config_custody["sha256"]),
        receiver=receiver,
        movable_layer=movable_layer,
        parent_bytes=parent_bytes,
        base_sha256=base_sha256,
        parameter_map=parameter_map,
        labels=labels,
        poses=poses,
        scorers=scorers,
    )
    source = _exact_verdict(
        candidate_id=candidate_id,
        label="source",
        root=root,
        receiver=receiver,
        movable_layer=movable_layer,
        parent_bytes=parent_bytes,
        base_sha256=base_sha256,
        parameter_map=parameter_map,
        q_xi=np.zeros((32, 6), dtype=np.int16),
        labels=labels,
        poses=poses,
        centers=centers,
        factors=factors,
        tube_radius=tube_radius,
        scorers=scorers,
    )
    final = _exact_verdict(
        candidate_id=candidate_id,
        label="final",
        root=root,
        receiver=receiver,
        movable_layer=movable_layer,
        parent_bytes=parent_bytes,
        base_sha256=base_sha256,
        parameter_map=parameter_map,
        q_xi=q_xi,
        labels=labels,
        poses=poses,
        centers=centers,
        factors=factors,
        tube_radius=tube_radius,
        scorers=scorers,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "candidate_id": candidate_id,
        "base_archive": {
            "path": binding["path"],
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
            "receiver_candidate": receiver.parsed.candidate,
            "exact_parseback": True,
        },
        "source_local_receiver": {
            "equation": "C(q;W)=clip_u8(W+P(q;W)-P(0;W))",
            "q_zero_parent_byte_identity": True,
            "counted_coordinate_quantum": "q=256 == one physical PC1 quantum",
            "receiver_composite_r_uint8": True,
            "deterministic_parseback": True,
        },
        "bounded_local_descent": {
            "descent_was_run": True,
            "schedule": "32-knot bit-reversal; six Pose axes; both signs",
            "candidate_evaluations": 384,
            "accepted_step_cap": 8,
            "accepted_step_count": len(accepted),
            "accepted_steps": accepted,
            "all_schedule_stage_checkpoints_preserved": True,
            "selection_only_local_batch": 4,
            "verdict_batch": 32,
            "verdict_n": 600,
        },
        "source_n600": source,
        "final_n600": final,
        "candidate_local_descent_rate": _rate_rows(source, final),
        "active_tube": {
            "tube_claim": True,
            "membership_claim": final["active_tube"]["all_pairs_inside"],
            "active_dimension_count": final["active_tube"]["active_dimension_count"],
            "dimension_slacks": final["active_tube"]["dimensions"],
            "full_quadratic": final["active_tube"],
        },
        "n600_joint_delta": float(final["advisory_action"]) - float(source["advisory_action"]),
        "n600_d_seg_delta": float(final["d_seg"]) - float(source["d_seg"]),
        "n600_d_pose_delta": float(final["d_pose"]) - float(source["d_pose"]),
        "verdict": (
            "MEASURED_ACTIVE_TUBE_MEMBER"
            if final["active_tube"]["all_pairs_inside"]
            else "MEASURED_OUTSIDE_ACTIVE_TUBE"
        ),
        "verdict_scope": (
            f"INSTANCE: {candidate_id} exact SHA-bound source under bounded "
            "source-local PC1 q=ONE_QUANTUM coordinate descent"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    receipt["content_sha256"] = sha256_bytes(canonical_bytes(receipt))
    _write_json(root / "active_tube_receipt.json", receipt)
    return receipt


def _rg3_harvest(
    config: Mapping[str, Any],
    config_custody: Mapping[str, Any],
    research_root: Path,
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    rebuilt = build_summary(
        checkpoint_roots=[Path(value) for value in config["checkpoint_roots"]],
        assignment_table_path=_resolve(artifacts["rg3_table"]["path"]),
        g3_path=_resolve(artifacts["g3_registry"]["path"]),
        rg2_assignment_path=_resolve(artifacts["rg2_assignment"]["path"]),
        rg3_assignment_path=_resolve(artifacts["rg3_assignment"]["path"]),
    )
    terminal = _read_json(_resolve(artifacts["rg3_summary"]["path"]))
    rebuilt_semantic = dict(rebuilt)
    terminal_semantic = dict(terminal)
    for value in (rebuilt_semantic, terminal_semantic):
        value.pop("summary_content_sha256", None)
        custody = dict(value["input_custody"])
        for key in ("assignment_table", "g3_registry", "rg2_assignment", "rg3_assignment"):
            custody[key] = dict(custody[key])
            custody[key].pop("path", None)
        value["input_custody"] = custody
    if rebuilt_semantic != terminal_semantic:
        raise RunnerError("RG3 aggregate rebuild differs beyond relocatable paths")
    exclusions = rg3_typed_exclusions(rebuilt)
    enriched = dict(rebuilt)
    enriched["rg4_obstruction_closure"] = {
        "schema": "ddm_rg4_rg3_top24_obstruction_closure.v1",
        "requested_missing_block_count": 25,
        "positive_closure_count": 0,
        "typed_exclusion_count": 25,
        "partial_coverage_proven": True,
        "proven_nonexcluded_coverage": True,
        "coverage_proven": False,
        "falsifier_fired": True,
        "typed_exclusions": exclusions,
        "verdict": ("FALSIFIED_ALL_25_POSITIVE_CLOSURES; EXHAUSTIVE_INSTANCE_OBSTRUCTIONS_PRESERVED"),
        "verdict_scope": (
            "INSTANCE: exact 25 pair/bucket blocks under production RG1/RG2/RG3 "
            "families and all already-measured signed magnitudes"
        ),
        "required_new_coordinate_families": sorted({row["derived_next_coordinate_family"] for row in exclusions}),
    }
    enriched.pop("summary_content_sha256", None)
    enriched["summary_content_sha256"] = sha256_bytes(canonical_bytes(enriched))
    summary_ref = _write_json(
        research_root / "ddm_rg4_receiver_support_summary.json",
        enriched,
    )
    table_bytes = _resolve(artifacts["rg3_table"]["path"]).read_bytes()
    table_ref = _atomic_bytes(
        research_root / "ms5_loader_assignment_table.json",
        table_bytes,
    )
    if table_ref["sha256"] != artifacts["rg3_table"]["sha256"]:
        raise RunnerError("MS5 loader table copy differs")
    ms2r_config = _read_json(_resolve(artifacts["ms2r_config"]["path"]))
    inputs = {name: _read_json(_resolve(binding["path"])) for name, binding in ms2r_config["inputs"].items()}
    inputs["rg3"] = enriched
    validate_ms2r_r3_sources(inputs)
    inner_jacobian = _read_json(_resolve(artifacts["inner_jacobian_status"]["path"]))
    if (
        inner_jacobian.get("schema") != "m1_band_inner_jacobian_secant_qp_status.v1"
        or inner_jacobian.get("realized_backbone_secants") != "ABSENT"
        or inner_jacobian.get("qp_receiver_closure") != "ABSENT"
    ):
        raise RunnerError("corrected inner-Jacobian status custody differs")
    receipt = {
        "schema": "ddm_rg4_rg3_25_block_coverage_receipt.v1",
        "typed_config": dict(config_custody),
        "rebuilt_terminal_aggregate": {
            "semantic_match_excluding_relocatable_paths": True,
            "terminal_summary_sha256": artifacts["rg3_summary"]["sha256"],
            "rebuilt_summary_content_sha256": rebuilt["summary_content_sha256"],
            "checkpoint_root_count": len(config["checkpoint_roots"]),
        },
        "ms5_loader_assignment_table": table_ref,
        "enriched_summary": summary_ref,
        "coverage": enriched["g3_top24_coverage"],
        "obstruction_closure": enriched["rg4_obstruction_closure"],
        "ms2r_r3_strict_join_replay": {
            "passed": True,
            "semantics_weakened": False,
            "coverage_proven": False,
            "missing_block_count": 25,
            "producer_rerun_eligible": False,
        },
        "corrected_inner_jacobian_bank": {
            "status_artifact": artifacts["inner_jacobian_status"],
            "first_order_vjp": inner_jacobian.get("first_order_vjp"),
            "realized_backbone_secants": inner_jacobian["realized_backbone_secants"],
            "qp_receiver_closure": inner_jacobian["qp_receiver_closure"],
            "formalization": inner_jacobian.get("formalization"),
            "claim": (
                "NO_GLOBAL_583_COMPLETION_CLAIM; the 25 instance verdicts rely "
                "on their actual receiver +/-quantum checkpoint measurements"
            ),
        },
        "measurement_reuse": {
            "already_settled_table_respected": True,
            "new_scorer_passes_for_rg3": 0,
            "reason": (
                "all signs and admissible magnitudes already have SHA-bound "
                "actual-receiver, composite-R, uint8 checkpoint custody"
            ),
        },
        "verdict": ("FALSIFIER_FIRED_25_OF_25_TYPED_INSTANCE_OBSTRUCTIONS; G3_COVERAGE_REMAINS_UNPROVEN"),
        "verdict_scope": enriched["rg4_obstruction_closure"]["verdict_scope"],
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    receipt["content_sha256"] = sha256_bytes(canonical_bytes(receipt))
    _write_json(research_root / "rg3_25_block_coverage_receipt.json", receipt)
    return receipt


def run(config_path: Path) -> dict[str, Any]:
    config, config_custody = _load_config(config_path.resolve(strict=True))
    output_root = Path(config["output_root"])
    research_root = _resolve(config["research_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    research_root.mkdir(parents=True, exist_ok=True)
    for tier in (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    ):
        if output_root.resolve() != tier and output_root.resolve().is_relative_to(tier.resolve()):
            break
    else:
        raise RunnerError("RG4 output root must use the SSD waterfall")
    disk = shutil.disk_usage(output_root)
    if disk.free < MIN_STORAGE_BYTES:
        raise RunnerError("REFUSE_RG4_STORAGE: SSD free bytes below 20 GiB")
    preflight_path = output_root / "00_preflight_receipt.json"
    if preflight_path.exists():
        preflight = _read_json(preflight_path)
        if (
            preflight.get("schema") != "ddm_rg4_g3_blocks_and_active_tube_preflight.v1"
            or preflight.get("run_id") != config["run_id"]
            or preflight.get("typed_config", {}).get("sha256") != config_custody["sha256"]
            or preflight.get("admission") is not True
        ):
            raise RunnerError("preserved RG4 preflight identity differs")
        _memory_receipt("RESUME")
    else:
        preflight = {
            "schema": "ddm_rg4_g3_blocks_and_active_tube_preflight.v1",
            "run_id": config["run_id"],
            "lane_id": config["lane_id"],
            "typed_config": config_custody,
            "storage": {
                "output_root": str(output_root),
                "free_bytes": disk.free,
                "required_free_bytes": MIN_STORAGE_BYTES,
                "cleanup": (
                    "camera tensors are chunk-local and released; immutable stage "
                    "checkpoints, verdict chunks, receipts, and final archives persist"
                ),
            },
            "memory": _memory_receipt("PREFLIGHT"),
            "governor": {
                "paid_dispatch": False,
                "training": False,
                "pair_count": 600,
                "verdict_batch": 32,
                "candidate_count": 2,
                "candidate_evaluations_each": 384,
            },
            "admission": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "main_landing_review_required": True,
        }
        _write_json(preflight_path, preflight)
    coverage = _rg3_harvest(config, config_custody, research_root)
    _configure_torch(config)
    target_path = _resolve(config["artifacts"]["target_cache"]["path"])
    labels = open_stored_npy_memmap(target_path, "lstars")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    if labels.shape != (600, 384, 512) or poses.shape != (600, 6):
        raise RunnerError("RG4 target cache geometry differs")
    metric = _read_json(_resolve(config["artifacts"]["pose_metric"]["path"]))
    rows = metric.get("rows")
    if (
        metric.get("schema") != "ddm_pose_metric_custody.v1"
        or metric.get("pair_count") != 600
        or metric.get("scorer_batch_size") != 32
        or metric.get("output_dimension") != 6
        or not isinstance(rows, list)
        or len(rows) != 600
    ):
        raise RunnerError("RG4 Pose metric custody differs")
    centers = np.asarray([row["center"] for row in rows], dtype=np.float64)
    factors = np.asarray([row["low_rank_factors"] for row in rows], dtype=np.float64)
    tube_radius = float(metric["tube_radius"])
    segnet, posenet, scorer_custody = _load_scorers(config["upstream_root"])
    menu = _read_json(_resolve(config["artifacts"]["menu1_config"]["path"]))
    if (
        scorer_custody["modules"]["sha256"] != menu.get("modules_sha256")
        or scorer_custody["segnet"]["sha256"] != menu.get("segnet_weights_sha256")
        or scorer_custody["posenet"]["sha256"] != menu.get("posenet_weights_sha256")
    ):
        raise RunnerError("RG4 frozen scorer custody differs from Menu1")
    parameter_map = _parameter_map(config)
    candidates = {}
    for candidate_id, binding_name in (
        ("c1_composed_line_current", "w_joint"),
        ("ws2_w_seg_138031", "w_seg"),
    ):
        candidate_receipt = _run_candidate(
            candidate_id=candidate_id,
            binding_name=binding_name,
            config=config,
            config_custody=config_custody,
            output_root=output_root,
            labels=labels,
            poses=poses,
            centers=centers,
            factors=factors,
            tube_radius=tube_radius,
            parameter_map=parameter_map,
            scorers=(segnet, posenet),
        )
        candidates[candidate_id] = candidate_receipt
        _write_json(
            research_root / f"{candidate_id}_active_tube_receipt.json",
            candidate_receipt,
        )
    final = {
        "schema": "ddm_rg4_g3_blocks_and_active_tube_run_receipt.v1",
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "coverage_receipt": coverage,
        "candidate_receipts": candidates,
        "scorer_custody": scorer_custody,
        "deterministic_algorithms": True,
        "resumability": {
            "search_stage_checkpoints_per_candidate": 33,
            "n600_chunks_per_candidate_state": 19,
            "all_stage_checkpoints_preserved": True,
            "resume_behavior": "rehash all inputs and reuse only identity-matching immutable stages",
        },
        "verdict": "MEASURED_RG3_OBSTRUCTIONS_AND_CANDIDATE_LOCAL_ACTIVE_TUBES",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    final["content_sha256"] = sha256_bytes(canonical_bytes(final))
    _write_json(research_root / "run_receipt.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "coverage_proven": receipt["coverage_receipt"]["coverage"]["coverage_proven"],
                "candidate_verdicts": {key: value["verdict"] for key, value in receipt["candidate_receipts"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
