#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the bounded, resumable PC1 solved-plane pose-descent smoke."""

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
from itertools import pairwise
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for _path in (_SRC, _REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    open_stored_npy_memmap,
)
from tac.optimization.ddm_pc1_pose_stream import (  # noqa: E402
    DDMPC1TrainableParameterMapV1,
    PC1PosePacketV1,
    build_counted_composition_archive,
    parse_counted_composition_archive,
    receive_pc1_camera_pairs,
    serialize_pc1_packet,
)
from tac.optimization.ddm_pc2_pose_descent import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    EVIDENCE_AXIS,
    POINTER,
    POSE_AXES,
    RECEIPT_SCHEMA,
    VERDICT_SCHEMA,
    PC2PoseDescentConfigV1,
    PC2PoseDescentError,
    bit_reversal_knot_order,
    canonical_bytes,
    constant_slope_horizon,
    fork_verdict,
    four_pair_batch_for_knot,
    realized_slope_row,
    score_domain_action,
    select_realized_candidate,
    sha256_bytes,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    receive_ws1_warm_start_archive,
)
from tools.measure_ddm_menu1_realized_flip_menu import _forward  # noqa: E402

MIN_AVAILABLE_BYTES = 20 * 1024**3
MIN_STORAGE_BYTES = 20 * 1024**3
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PC2PoseDescentError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_bytes(dict(payload)))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PC2PoseDescentError(f"JSON artifact unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise PC2PoseDescentError(f"JSON artifact must contain one object: {path}")
    return value


def _available_memory() -> tuple[int, int]:
    try:
        import psutil
    except ImportError as exc:
        raise PC2PoseDescentError("PC2 exact-verdict preflight requires psutil") from exc
    value = psutil.virtual_memory()
    return int(value.total), int(value.available)


def _memory_receipt(*, stage: str) -> dict[str, Any]:
    total, available = _available_memory()
    if available < MIN_AVAILABLE_BYTES:
        raise PC2PoseDescentError(f"REFUSE_PC2_MEMORY_{stage}: available {available} < {MIN_AVAILABLE_BYTES}")
    return {
        "stage": stage,
        "total_bytes": total,
        "available_bytes": available,
        "required_available_bytes": MIN_AVAILABLE_BYTES,
        "admission": True,
        "source": "psutil.virtual_memory",
    }


def _output_tier(output_root: Path) -> str:
    resolved = output_root.resolve()
    for tier_value in (
        "/Volumes/VertigoDataTier/pact",
        "/Volumes/APDataStore/pact",
    ):
        tier = Path(tier_value).resolve()
        if resolved != tier and resolved.is_relative_to(tier):
            return str(tier)
    raise PC2PoseDescentError("PC2 output root must use the SSD waterfall")


def _configure_torch(config: PC2PoseDescentConfigV1) -> int:
    import torch

    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    observed = int(torch.get_num_threads())
    if observed != config.torch_threads:
        raise PC2PoseDescentError(f"REFUSE_PC2_TORCH_THREADS: {observed} != {config.torch_threads}")
    return observed


def _load_cpu_frozen_scorers(upstream_root: str) -> tuple[Any, Any]:
    from safetensors.torch import load_file

    upstream = Path(upstream_root)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import modules as upstream_modules

    modules_path = upstream / "modules.py"
    if Path(upstream_modules.__file__).resolve() != modules_path.resolve():
        raise PC2PoseDescentError("frozen scorer imported a non-custodied modules.py")
    segnet = upstream_modules.SegNet().eval().to("cpu")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    segnet.load_state_dict(load_file(str(upstream_modules.segnet_sd_path), device="cpu"))
    posenet.load_state_dict(load_file(str(upstream_modules.posenet_sd_path), device="cpu"))
    for scorer in (segnet, posenet):
        for parameter in scorer.parameters():
            parameter.requires_grad = False
    return segnet, posenet


def _validate_semantic_sources(
    config: PC2PoseDescentConfigV1,
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pc1 = _read_json(config.source_artifacts["pc1_admission"].resolve(config.repo_root))
    ws4 = _read_json(config.source_artifacts["ws4_arbitration"].resolve(config.repo_root))
    menu = _read_json(config.source_artifacts["menu1_config"].resolve(config.repo_root))
    j7 = _read_json(config.source_artifacts["j7_precedent"].resolve(config.repo_root))
    parent = config.source_artifacts["w_joint_step50"]
    ws4_parent = ws4.get("candidate_custody", {}).get("W_joint_step50_live", {})
    if (
        pc1.get("schema") != "ddm_pc1_pose_stream_admission.v1"
        or pc1.get("admission", {}).get("descent_was_run") is not False
        or pc1.get("parameter_map", {}).get("descent_trainable") is not True
        or pc1.get("parameter_map", {}).get("coordinate_count") != 320
    ):
        raise PC2PoseDescentError("PC1 admission semantics differ")
    if (
        ws4.get("selected_warm_start") != "W_joint_step50_live"
        or ws4_parent.get("sha256") != parent.sha256
        or ws4_parent.get("bytes") != parent.bytes
        or ws4_parent.get("parameter_shadow") != "live_resume_state"
    ):
        raise PC2PoseDescentError("ws4 selected warm-start custody differs")
    target = config.source_artifacts["target_cache"]
    if (
        menu.get("target_cache_path") != target.path
        or menu.get("target_cache_sha256") != target.sha256
        or menu.get("target_cache_bytes") != target.bytes
        or menu.get("upstream_root") != config.upstream_root
        or menu.get("scorer_threads") != config.torch_threads
    ):
        raise PC2PoseDescentError("frozen scorer/menu custody differs")
    upstream = Path(config.upstream_root)
    scorer_files = {
        "modules": upstream / "modules.py",
        "segnet": upstream / "models" / "segnet.safetensors",
        "posenet": upstream / "models" / "posenet.safetensors",
    }
    scorer_custody = {}
    for name, path in scorer_files.items():
        byte_count, digest = _sha256_file(path)
        scorer_custody[name] = {
            "path": str(path),
            "bytes": byte_count,
            "sha256": digest,
        }
    if (
        scorer_custody["modules"]["sha256"] != menu.get("modules_sha256")
        or scorer_custody["segnet"]["sha256"] != menu.get("segnet_weights_sha256")
        or scorer_custody["posenet"]["sha256"] != menu.get("posenet_weights_sha256")
    ):
        raise PC2PoseDescentError("frozen scorer bytes differ from Menu1 custody")
    if (
        j7.get("schema") != "ddm_j7_366_fire_readiness_receipt.v1"
        or j7.get("research_only") is not True
        or j7.get("execution_allowed") is not False
    ):
        raise PC2PoseDescentError("J7 bounded-run precedent semantics differ")
    return {
        "bindings": dict(bindings),
        "pc1_admission": {
            "descent_was_run": False,
            "coordinate_count": 320,
            "xi_scales": list(pc1["parameter_map"]["xi_scales"]),
            "residual_scale": float(pc1["parameter_map"]["residual_scale"]),
        },
        "ws4": {
            "selected_warm_start": ws4["selected_warm_start"],
            "parent_archive_sha256": parent.sha256,
            "parent_archive_bytes": parent.bytes,
            "parameter_shadow": ws4_parent["parameter_shadow"],
        },
        "compile_refuse": {
            "score_domain_loss": config.score_domain_loss,
            "pose_marginal_weight_law": config.pose_marginal_weight_law,
            "pose_objective_weight": config.pose_objective_weight,
            "decision": (
                "score_domain_loss XOR PoseMarginalWeightLaw; selected exact "
                "score-domain sqrt term with static w_pose=1"
            ),
        },
        "j7_governor_precedent": {
            "schema": j7["schema"],
            "verdict": j7["verdict"],
            "verdict_scope": j7["verdict_scope"],
            "execution_allowed_by_precedent": False,
            "application": (
                "precedent only: PC2 has separate delegated authority and remains "
                "bounded to 10-20 accepted local advisory steps"
            ),
        },
        "scorer_custody": scorer_custody,
    }


def _preflight(config: PC2PoseDescentConfigV1) -> dict[str, Any]:
    output_root = Path(config.output_root)
    preflight_path = output_root / "00_preflight_receipt.json"
    if preflight_path.exists():
        receipt = _read_json(preflight_path)
        if (
            receipt.get("schema") != "ddm_pc2_pose_descent_preflight.v1"
            or receipt.get("typed_config_hash") != config.typed_hash()
            or receipt.get("admission") is not True
        ):
            raise PC2PoseDescentError("preserved PC2 preflight differs")
        receipt["fresh_resume_memory"] = _memory_receipt(stage="RESUME")
        return receipt
    if Path(sys.executable).absolute() != Path(config.own_python).absolute():
        raise PC2PoseDescentError(f"PC2 must run in owned venv {config.own_python}; got {sys.executable}")
    started = time.monotonic()
    bindings = config.validate_all_bindings()
    source_semantics = _validate_semantic_sources(config, bindings)
    output_root.mkdir(parents=True, exist_ok=True)
    tier = _output_tier(output_root)
    disk = shutil.disk_usage(output_root)
    if disk.free < MIN_STORAGE_BYTES:
        raise PC2PoseDescentError("REFUSE_PC2_STORAGE: SSD free bytes below 20 GiB")
    memory = _memory_receipt(stage="PREFLIGHT")
    observed_threads = _configure_torch(config)
    parent_bytes = config.source_artifacts["w_joint_step50"].validate_bytes(config.repo_root)
    parent_receiver = receive_ws1_warm_start_archive(parent_bytes)
    if (
        parent_receiver.archive != parent_bytes
        or parent_receiver.parsed.exact_reemit() != parent_bytes
        or parent_receiver.parsed.candidate != "W_joint"
    ):
        raise PC2PoseDescentError("W_joint step50 receiver parse/re-emit differs")
    receipt = {
        "schema": "ddm_pc2_pose_descent_preflight.v1",
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "typed_config_hash": config.typed_hash(),
        "source_semantics": source_semantics,
        "memory": memory,
        "storage": {
            "tier": tier,
            "output_root": str(output_root),
            "free_bytes": disk.free,
            "required_free_bytes": MIN_STORAGE_BYTES,
            "cleanup": (
                "proposal cameras are process-local and released; immutable JSON "
                "accepted-step checkpoints, exact-verdict chunks, final archive, "
                "and receipt are preserved on the SSD"
            ),
        },
        "governor": {
            "train_batch": config.train_batch,
            "verdict_batch": config.verdict_batch,
            "target_accepted_steps": config.target_accepted_steps,
            "maximum_candidate_evaluations": config.maximum_candidate_evaluations,
            "proposal_quanta": list(config.proposal_quanta),
            "paid_dispatch": False,
            "contest_eval": False,
            "scope": "BOUNDED_LOCAL_ADVISORY_ONLY",
        },
        "deterministic_algorithms": True,
        "torch_threads": observed_threads,
        "elapsed_seconds": time.monotonic() - started,
        "admission": True,
        "execution_allowed_by_this_receipt": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_review_required": True,
    }
    _write_json(preflight_path, receipt)
    return receipt


def _parameter_map(
    config: PC2PoseDescentConfigV1,
) -> DDMPC1TrainableParameterMapV1:
    pc1 = _read_json(config.source_artifacts["pc1_admission"].resolve(config.repo_root))
    row = pc1["parameter_map"]
    return DDMPC1TrainableParameterMapV1(
        pair_count=config.pair_count,
        knot_count=int(row["coordinate_schema"]["knot_count"]),
        xi_scales=tuple(float(value) for value in row["xi_scales"]),
        residual_scale=float(row["residual_scale"]),
    )


def _packet_from_q(
    parameter_map: DDMPC1TrainableParameterMapV1,
    q_xi: np.ndarray,
) -> PC1PosePacketV1:
    value = np.asarray(q_xi)
    if value.shape != (parameter_map.knot_count, 6):
        raise PC2PoseDescentError("PC2 q_xi geometry differs")
    return PC1PosePacketV1(
        active=True,
        pair_count=parameter_map.pair_count,
        xi_scales=parameter_map.xi_scales,
        residual_scale=parameter_map.residual_scale,
        q_xi=np.ascontiguousarray(value, dtype="<i2"),
        q_luma_phase=np.zeros((parameter_map.knot_count, 4), dtype=np.int8),
    )


def _movable_layer(receiver: Any) -> Any:
    try:
        return next(layer for layer in receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise PC2PoseDescentError("W_joint step50 has no Movable layer") from exc


def _render_candidate(
    *,
    receiver: Any,
    movable_layer: Any,
    packet: PC1PosePacketV1,
    pair_ids: Sequence[int],
    parent_camera: np.ndarray | None = None,
    movable_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = tuple(int(value) for value in pair_ids)
    if parent_camera is None:
        parent_camera = receiver.render_camera_pairs(ids)
    if movable_masks is None:
        movable_masks = np.stack(
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
    candidate = receive_pc1_camera_pairs(
        parent_camera=parent_camera,
        packet=packet,
        pair_ids=ids,
        movable_masks=movable_masks,
    )
    return candidate, parent_camera, movable_masks


def _measure_camera(
    *,
    camera: np.ndarray,
    pair_ids: Sequence[int],
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
    archive_bytes: int,
) -> dict[str, Any]:
    cells, pose6 = _forward(scorers[0], scorers[1], camera)
    ids = tuple(int(value) for value in pair_ids)
    target = np.asarray(labels[list(ids)], dtype=np.uint8)
    target_pose = np.asarray(poses[list(ids)], dtype=np.float64)
    errors = int(np.count_nonzero(cells != target))
    sites = int(cells.size)
    pose_sse = float(np.square(pose6 - target_pose).sum(dtype=np.float64))
    pose_coordinates = int(pose6.size)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    return {
        "pair_ids": list(ids),
        "errors": errors,
        "sites": sites,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "advisory_action": score_domain_action(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
        ),
        "camera_sha256": sha256_bytes(camera.tobytes(order="C")),
    }


def _composition_archive(
    *,
    parent_bytes: bytes,
    parent_sha256: str,
    packet: PC1PosePacketV1,
) -> bytes:
    archive = build_counted_composition_archive(
        parent_archive=parent_bytes,
        parent_sha256=parent_sha256,
        packet=packet,
    )
    parsed_parent, parsed_packet, _manifest = parse_counted_composition_archive(archive)
    if parsed_parent != parent_bytes or serialize_pc1_packet(parsed_packet) != serialize_pc1_packet(packet):
        raise PC2PoseDescentError("PC1 composition parse-back differs")
    return archive


def _checkpoint_path(output_root: Path, accepted_steps: int) -> Path:
    return output_root / "checkpoints" / f"accepted_{accepted_steps:03d}.json"


def _save_checkpoint(
    *,
    config: PC2PoseDescentConfigV1,
    q_xi: np.ndarray,
    accepted_steps: Sequence[Mapping[str, Any]],
    candidate_evaluations: int,
    schedule_cursor: int,
) -> dict[str, Any]:
    packet = _packet_from_q(_parameter_map(config), q_xi)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": config.run_id,
        "typed_config_hash": config.typed_hash(),
        "accepted_step_count": len(accepted_steps),
        "candidate_evaluations": candidate_evaluations,
        "schedule_cursor": schedule_cursor,
        "q_xi": np.asarray(q_xi, dtype=np.int16).tolist(),
        "q_luma_phase": np.zeros((len(q_xi), 4), dtype=np.int8).tolist(),
        "packet_sha256": sha256_bytes(serialize_pc1_packet(packet)),
        "accepted_steps": list(accepted_steps),
        "status": "PRESERVED_COMPLETE_STATE",
        "all_prior_accepted_stage_checkpoints_preserved": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    path = _checkpoint_path(Path(config.output_root), len(accepted_steps))
    _write_json(path, payload)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def _resume_checkpoint(
    config: PC2PoseDescentConfigV1,
) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    output_root = Path(config.output_root)
    parameter_map = _parameter_map(config)
    latest: dict[str, Any] | None = None
    gap = False
    for accepted in range(config.target_accepted_steps + 1):
        path = _checkpoint_path(output_root, accepted)
        if not path.exists():
            gap = True
            continue
        if gap:
            raise PC2PoseDescentError("PC2 accepted-step checkpoints contain a gap")
        row = _read_json(path)
        if (
            row.get("schema") != CHECKPOINT_SCHEMA
            or row.get("typed_config_hash") != config.typed_hash()
            or row.get("accepted_step_count") != accepted
            or row.get("status") != "PRESERVED_COMPLETE_STATE"
            or len(row.get("accepted_steps", [])) != accepted
        ):
            raise PC2PoseDescentError(f"PC2 checkpoint identity differs: {path}")
        q_xi = np.asarray(row["q_xi"], dtype=np.int16)
        packet = _packet_from_q(parameter_map, q_xi)
        if sha256_bytes(serialize_pc1_packet(packet)) != row.get("packet_sha256"):
            raise PC2PoseDescentError(f"PC2 checkpoint packet differs: {path}")
        latest = row
    if latest is None:
        q_xi = np.zeros((parameter_map.knot_count, 6), dtype=np.int16)
        _save_checkpoint(
            config=config,
            q_xi=q_xi,
            accepted_steps=(),
            candidate_evaluations=0,
            schedule_cursor=0,
        )
        return q_xi, [], 0, 0
    return (
        np.asarray(latest["q_xi"], dtype=np.int16),
        [dict(row) for row in latest["accepted_steps"]],
        int(latest["candidate_evaluations"]),
        int(latest["schedule_cursor"]),
    )


def _exact_chunk_path(output_root: Path, accepted_step: int, start: int, stop: int) -> Path:
    return output_root / "verdicts" / f"accepted_{accepted_step:03d}" / f"chunk_{start:04d}_{stop:04d}.json"


def _exact_n600_verdict(
    *,
    config: PC2PoseDescentConfigV1,
    accepted_step: int,
    packet: PC1PosePacketV1,
    receiver: Any,
    movable_layer: Any,
    parent_bytes: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
) -> dict[str, Any]:
    output_root = Path(config.output_root)
    verdict_path = output_root / "verdicts" / f"accepted_{accepted_step:03d}" / "n600.json"
    archive = _composition_archive(
        parent_bytes=parent_bytes,
        parent_sha256=config.source_artifacts["w_joint_step50"].sha256,
        packet=packet,
    )
    archive_sha = sha256_bytes(archive)
    packet_sha = sha256_bytes(serialize_pc1_packet(packet))
    if verdict_path.exists():
        value = _read_json(verdict_path)
        if (
            value.get("schema") != VERDICT_SCHEMA
            or value.get("archive_sha256") != archive_sha
            or value.get("accepted_step") != accepted_step
        ):
            raise PC2PoseDescentError("preserved PC2 n600 verdict differs")
        return value
    memory = _memory_receipt(stage=f"N600_ACCEPTED_{accepted_step:03d}")
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, config.pair_count, config.verdict_batch):
        stop = min(start + config.verdict_batch, config.pair_count)
        path = _exact_chunk_path(output_root, accepted_step, start, stop)
        if path.exists():
            row = _read_json(path)
            if (
                row.get("packet_sha256") != packet_sha
                or row.get("archive_sha256") != archive_sha
                or row.get("pair_range") != [start, stop]
            ):
                raise PC2PoseDescentError(f"preserved PC2 exact chunk differs: {path}")
            rows.append(row)
            continue
        ids = tuple(range(start, stop))
        camera, _parent, _masks = _render_candidate(
            receiver=receiver,
            movable_layer=movable_layer,
            packet=packet,
            pair_ids=ids,
        )
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        cells, pose6 = _forward(scorers[0], scorers[1], camera)
        errors = int(np.count_nonzero(cells != target))
        sites = int(cells.size)
        pose_sse = float(np.square(pose6 - target_pose).sum(dtype=np.float64))
        pose_coordinates = int(pose6.size)
        per_class = {
            class_name: {
                "errors": int(np.count_nonzero((cells != target) & (target == class_id))),
                "sites": int(np.count_nonzero(target == class_id)),
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        }
        row = {
            "schema": "ddm_pc2_pose_descent_n600_chunk.v1",
            "accepted_step": accepted_step,
            "pair_range": [start, stop],
            "packet_sha256": packet_sha,
            "archive_sha256": archive_sha,
            "errors": errors,
            "sites": sites,
            "pose_squared_error_sum": pose_sse,
            "pose_coordinates": pose_coordinates,
            "camera_sha256": sha256_bytes(camera.tobytes(order="C")),
            "per_class": per_class,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        _write_json(path, row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "exact_n600_chunk",
                    "accepted_step": accepted_step,
                    "pair_range": [start, stop],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del camera, target, target_pose, cells, pose6
        gc.collect()
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    per_class = {
        class_name: {
            "class_id": class_id,
            "errors": sum(int(row["per_class"][class_name]["errors"]) for row in rows),
            "sites": sum(int(row["per_class"][class_name]["sites"]) for row in rows),
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    for row in per_class.values():
        row["d_seg"] = row["errors"] / row["sites"]
    if errors != sum(int(row["errors"]) for row in per_class.values()):
        raise PC2PoseDescentError("PC2 global/per-class error totals differ")
    verdict = {
        "schema": VERDICT_SCHEMA,
        "accepted_step": accepted_step,
        "num_pairs": config.pair_count,
        "batch_size": config.verdict_batch,
        "maximum_rgb_chunks_resident": 1,
        "packet_sha256": packet_sha,
        "archive_bytes": len(archive),
        "archive_sha256": archive_sha,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "errors": errors,
        "sites": sites,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "advisory_action": score_domain_action(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=len(archive),
        ),
        "per_class": per_class,
        "memory_preflight": memory,
        "chunk_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    _write_json(verdict_path, verdict)
    archive_path = output_root / "archives" / f"accepted_{accepted_step:03d}_pc1.zip.receipt-bytes"
    _atomic_write(archive_path, archive)
    print(
        json.dumps(
            {
                "stage": "exact_n600",
                "accepted_step": accepted_step,
                "d_seg": d_seg,
                "d_pose": d_pose,
                "archive_bytes": len(archive),
                "advisory_action": verdict["advisory_action"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return verdict


def _proposal_row(
    *,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    coordinate_id: str,
    quantum: int,
    direction: int,
    receiver_visible: bool,
) -> dict[str, Any]:
    return {
        "coordinate_id": coordinate_id,
        "quantum": quantum,
        "direction": direction,
        "pair_ids": candidate["pair_ids"],
        "receiver_visible": receiver_visible,
        "d_seg": candidate["d_seg"],
        "d_pose": candidate["d_pose"],
        "archive_bytes": candidate["archive_bytes"],
        "advisory_action": candidate["advisory_action"],
        "seg_delta": float(candidate["d_seg"]) - float(current["d_seg"]),
        "pose_delta": float(candidate["d_pose"]) - float(current["d_pose"]),
        "rate_bytes_delta": int(candidate["archive_bytes"]) - int(current["archive_bytes"]),
        "joint_delta": float(candidate["advisory_action"]) - float(current["advisory_action"]),
        "camera_sha256": candidate["camera_sha256"],
    }


def _run(config: PC2PoseDescentConfigV1) -> dict[str, Any]:
    output_root = Path(config.output_root)
    final_path = output_root / "ddm_pc2_pose_descent_smoke_receipt.json"
    preflight = _preflight(config)
    if final_path.exists():
        final = _read_json(final_path)
        if final.get("schema") != RECEIPT_SCHEMA or final.get("typed_config_hash") != config.typed_hash():
            raise PC2PoseDescentError("completed PC2 receipt differs")
        return final
    _configure_torch(config)
    parent_bytes = config.source_artifacts["w_joint_step50"].validate_bytes(config.repo_root)
    parent_receiver = receive_ws1_warm_start_archive(parent_bytes)
    movable_layer = _movable_layer(parent_receiver)
    parameter_map = _parameter_map(config)
    target_path = config.source_artifacts["target_cache"].resolve(config.repo_root)
    labels = open_stored_npy_memmap(target_path, "lstars")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    if labels.shape != (600, 384, 512) or poses.shape != (600, 6):
        raise PC2PoseDescentError("PC2 target cache geometry differs")
    scorers = _load_cpu_frozen_scorers(config.upstream_root)
    q_xi, accepted_rows, candidate_evaluations, schedule_cursor = _resume_checkpoint(config)
    exact_verdicts: dict[int, dict[str, Any]] = {}
    for exact_step in config.exact_verdict_steps:
        path = output_root / "verdicts" / f"accepted_{exact_step:03d}" / "n600.json"
        if path.exists() and exact_step <= len(accepted_rows):
            exact_verdicts[exact_step] = _exact_n600_verdict(
                config=config,
                accepted_step=exact_step,
                packet=(
                    _packet_from_q(
                        parameter_map,
                        np.asarray(
                            _read_json(_checkpoint_path(output_root, exact_step))["q_xi"],
                            dtype=np.int16,
                        ),
                    )
                ),
                receiver=parent_receiver,
                movable_layer=movable_layer,
                parent_bytes=parent_bytes,
                labels=labels,
                poses=poses,
                scorers=scorers,
            )
    if 0 not in exact_verdicts:
        zero_checkpoint = _read_json(_checkpoint_path(output_root, 0))
        exact_verdicts[0] = _exact_n600_verdict(
            config=config,
            accepted_step=0,
            packet=_packet_from_q(
                parameter_map,
                np.asarray(zero_checkpoint["q_xi"], dtype=np.int16),
            ),
            receiver=parent_receiver,
            movable_layer=movable_layer,
            parent_bytes=parent_bytes,
            labels=labels,
            poses=poses,
            scorers=scorers,
        )

    order = bit_reversal_knot_order(parameter_map.knot_count)
    started = time.monotonic()
    stop_reason = "TARGET_ACCEPTED_STEPS_REACHED"
    while len(accepted_rows) < config.target_accepted_steps:
        if candidate_evaluations >= config.maximum_candidate_evaluations:
            stop_reason = "GOVERNOR_MAXIMUM_CANDIDATE_EVALUATIONS"
            break
        knot_id = order[schedule_cursor % len(order)]
        pair_ids = four_pair_batch_for_knot(
            knot_id,
            knot_count=parameter_map.knot_count,
            pair_count=config.pair_count,
        )
        current_packet = _packet_from_q(parameter_map, q_xi)
        current_archive = _composition_archive(
            parent_bytes=parent_bytes,
            parent_sha256=config.source_artifacts["w_joint_step50"].sha256,
            packet=current_packet,
        )
        current_camera, parent_camera, movable_masks = _render_candidate(
            receiver=parent_receiver,
            movable_layer=movable_layer,
            packet=current_packet,
            pair_ids=pair_ids,
        )
        current = _measure_camera(
            camera=current_camera,
            pair_ids=pair_ids,
            labels=labels,
            poses=poses,
            scorers=scorers,
            archive_bytes=len(current_archive),
        )
        selected: Mapping[str, Any] | None = None
        selected_q: np.ndarray | None = None
        evaluated_rows: list[dict[str, Any]] = []
        for quantum in config.proposal_quanta:
            quantum_rows: list[dict[str, Any]] = []
            quantum_states: dict[str, np.ndarray] = {}
            if candidate_evaluations + 12 > config.maximum_candidate_evaluations:
                stop_reason = "GOVERNOR_MAXIMUM_CANDIDATE_EVALUATIONS"
                break
            for axis_index, axis_name in enumerate(POSE_AXES):
                for direction in (-1, 1):
                    proposal_q = np.asarray(q_xi, dtype=np.int32).copy()
                    proposed = int(proposal_q[knot_id, axis_index]) + (direction * quantum)
                    coordinate_id = (
                        f"ddm.pc1.knot.{knot_id:03d}.xi.{axis_name}.{'plus' if direction > 0 else 'minus'}{quantum}"
                    )
                    candidate_evaluations += 1
                    if not np.iinfo(np.int16).min <= proposed <= np.iinfo(np.int16).max:
                        row = {
                            "coordinate_id": coordinate_id,
                            "quantum": quantum,
                            "direction": direction,
                            "pair_ids": list(pair_ids),
                            "receiver_visible": False,
                            "d_seg": current["d_seg"],
                            "d_pose": current["d_pose"],
                            "archive_bytes": current["archive_bytes"],
                            "advisory_action": current["advisory_action"],
                            "seg_delta": 0.0,
                            "pose_delta": 0.0,
                            "rate_bytes_delta": 0,
                            "joint_delta": 0.0,
                            "reason": "INT16_RANGE_REFUSE",
                        }
                        quantum_rows.append(row)
                        continue
                    proposal_q[knot_id, axis_index] = proposed
                    proposal_q_i16 = proposal_q.astype(np.int16)
                    proposal_packet = _packet_from_q(parameter_map, proposal_q_i16)
                    proposal_archive = _composition_archive(
                        parent_bytes=parent_bytes,
                        parent_sha256=config.source_artifacts["w_joint_step50"].sha256,
                        packet=proposal_packet,
                    )
                    candidate_camera, _parent, _masks = _render_candidate(
                        receiver=parent_receiver,
                        movable_layer=movable_layer,
                        packet=proposal_packet,
                        pair_ids=pair_ids,
                        parent_camera=parent_camera,
                        movable_masks=movable_masks,
                    )
                    visible = not np.array_equal(candidate_camera, current_camera)
                    if visible:
                        measured = _measure_camera(
                            camera=candidate_camera,
                            pair_ids=pair_ids,
                            labels=labels,
                            poses=poses,
                            scorers=scorers,
                            archive_bytes=len(proposal_archive),
                        )
                    else:
                        measured = {
                            **current,
                            "archive_bytes": len(proposal_archive),
                            "advisory_action": score_domain_action(
                                d_seg=float(current["d_seg"]),
                                d_pose=float(current["d_pose"]),
                                archive_bytes=len(proposal_archive),
                            ),
                            "camera_sha256": current["camera_sha256"],
                        }
                    row = _proposal_row(
                        current=current,
                        candidate=measured,
                        coordinate_id=coordinate_id,
                        quantum=quantum,
                        direction=direction,
                        receiver_visible=visible,
                    )
                    quantum_rows.append(row)
                    quantum_states[coordinate_id] = proposal_q_i16
                    del candidate_camera
                gc.collect()
            evaluated_rows.extend(quantum_rows)
            selected = select_realized_candidate(quantum_rows)
            if selected is not None:
                selected_q = quantum_states[str(selected["coordinate_id"])]
                break
        schedule_cursor += 1
        search_path = output_root / "telemetry" / f"search_cursor_{schedule_cursor:04d}.json"
        search_receipt = {
            "schema": "ddm_pc2_pose_descent_search.v1",
            "schedule_cursor": schedule_cursor,
            "knot_id": knot_id,
            "pair_ids": list(pair_ids),
            "current": current,
            "proposal_rows": evaluated_rows,
            "selected_coordinate_id": (None if selected is None else selected["coordinate_id"]),
            "candidate_evaluations_after": candidate_evaluations,
            "score_domain_loss": True,
            "pose_objective_weight": 1.0,
            "pose_marginal_weight_law": False,
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        }
        _write_json(search_path, search_receipt)
        if selected is None or selected_q is None:
            if schedule_cursor % len(order) == 0 and len(accepted_rows) == 0:
                stop_reason = "COMPLETE_32_KNOT_SWEEP_NO_REALIZED_POSE_AND_JOINT_DESCENT"
                break
            continue
        q_xi = selected_q
        accepted_row = {
            "accepted_step": len(accepted_rows) + 1,
            "schedule_cursor": schedule_cursor,
            "knot_id": knot_id,
            "pair_ids": list(pair_ids),
            "coordinate_id": selected["coordinate_id"],
            "quantum": selected["quantum"],
            "direction": selected["direction"],
            "local_before": current,
            "local_after": {
                "d_seg": selected["d_seg"],
                "d_pose": selected["d_pose"],
                "archive_bytes": selected["archive_bytes"],
                "advisory_action": selected["advisory_action"],
            },
            "local_delta": {
                "d_seg": selected["seg_delta"],
                "d_pose": selected["pose_delta"],
                "archive_bytes": selected["rate_bytes_delta"],
                "advisory_action": selected["joint_delta"],
            },
            "acceptance": ("STRICT_REALIZED_4PAIR_POSE_DESCENT_AND_SCORE_DOMAIN_JOINT_DELTA_NEGATIVE"),
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        }
        accepted_rows.append(accepted_row)
        checkpoint = _save_checkpoint(
            config=config,
            q_xi=q_xi,
            accepted_steps=accepted_rows,
            candidate_evaluations=candidate_evaluations,
            schedule_cursor=schedule_cursor,
        )
        accepted_row["checkpoint"] = checkpoint
        print(
            json.dumps(
                {
                    "stage": "accepted",
                    "accepted_step": len(accepted_rows),
                    "coordinate_id": selected["coordinate_id"],
                    "local_d_pose_delta": selected["pose_delta"],
                    "local_joint_delta": selected["joint_delta"],
                    "candidate_evaluations": candidate_evaluations,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if len(accepted_rows) in config.exact_verdict_steps:
            exact_verdicts[len(accepted_rows)] = _exact_n600_verdict(
                config=config,
                accepted_step=len(accepted_rows),
                packet=_packet_from_q(parameter_map, q_xi),
                receiver=parent_receiver,
                movable_layer=movable_layer,
                parent_bytes=parent_bytes,
                labels=labels,
                poses=poses,
                scorers=scorers,
            )
    final_step = len(accepted_rows)
    final_packet = _packet_from_q(parameter_map, q_xi)
    if final_step not in exact_verdicts:
        exact_verdicts[final_step] = _exact_n600_verdict(
            config=config,
            accepted_step=final_step,
            packet=final_packet,
            receiver=parent_receiver,
            movable_layer=movable_layer,
            parent_bytes=parent_bytes,
            labels=labels,
            poses=poses,
            scorers=scorers,
        )
    start_verdict = exact_verdicts[0]
    final_verdict = exact_verdicts[final_step]
    exact_steps = sorted(exact_verdicts)
    slope_rows = []
    for start_step, end_step in pairwise(exact_steps):
        if end_step <= start_step:
            continue
        slope_rows.append(
            {
                "window": [start_step, end_step],
                **realized_slope_row(
                    start=exact_verdicts[start_step],
                    end=exact_verdicts[end_step],
                    accepted_steps=end_step - start_step,
                    critical_ratio=config.critical_ratio,
                ),
            }
        )
    if final_step > 0 and (not slope_rows or exact_steps != [0, final_step]):
        slope_rows.append(
            {
                "window": [0, final_step],
                "aggregate": True,
                **realized_slope_row(
                    start=start_verdict,
                    end=final_verdict,
                    accepted_steps=final_step,
                    critical_ratio=config.critical_ratio,
                ),
            }
        )
    verdict, verdict_scope = fork_verdict(start=start_verdict, end=final_verdict)
    horizon = (
        None
        if final_step == 0
        else constant_slope_horizon(
            start_d_pose=float(start_verdict["d_pose"]),
            end_d_pose=float(final_verdict["d_pose"]),
            accepted_steps=final_step,
            target_d_pose=config.target_d_pose,
        )
    )
    final_archive = _composition_archive(
        parent_bytes=parent_bytes,
        parent_sha256=config.source_artifacts["w_joint_step50"].sha256,
        packet=final_packet,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "delegation_checkpoint_key": config.delegation_checkpoint_key,
        "typed_config_hash": config.typed_hash(),
        "preflight": {
            "path": str(output_root / "00_preflight_receipt.json"),
            "sha256": sha256_bytes((output_root / "00_preflight_receipt.json").read_bytes()),
            "admission": preflight["admission"],
        },
        "objective": {
            "score_domain_loss": True,
            "pose_objective_weight": 1.0,
            "pose_marginal_weight_law": False,
            "compile_refuse_decision": ("PoseMarginalWeightLaw XOR score_domain_loss; score-domain selected"),
            "local_acceptance": ("strict realized 4-pair pose descent AND exact score-domain joint delta negative"),
        },
        "governor": {
            "target_accepted_steps": config.target_accepted_steps,
            "accepted_steps": final_step,
            "candidate_evaluations": candidate_evaluations,
            "maximum_candidate_evaluations": config.maximum_candidate_evaluations,
            "stop_reason": stop_reason,
            "train_batch": config.train_batch,
            "verdict_batch": config.verdict_batch,
            "paid_dispatch": False,
        },
        "parent": {
            "role": "ws4 W_joint_step50_live",
            "archive_bytes": len(parent_bytes),
            "archive_sha256": sha256_bytes(parent_bytes),
            "parameter_shadow": "live_resume_state",
        },
        "accepted_step_rows": accepted_rows,
        "exact_verdicts": {str(step): exact_verdicts[step] for step in exact_steps},
        "slope_rows": slope_rows,
        "horizon_to_2_94e_5": horizon,
        "final_archive": {
            "path": str(output_root / "archives" / f"accepted_{final_step:03d}_pc1.zip.receipt-bytes"),
            "bytes": len(final_archive),
            "sha256": sha256_bytes(final_archive),
            "parseback_exact": True,
        },
        "verdict": verdict,
        "verdict_scope": verdict_scope,
        "fork": (
            "PC1_DESCENT_STAGE"
            if verdict == "PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE"
            else ("XI_ADVECTION_PREDICT_STAGE_COMPOSITION_CHANGE_601_605_LINEAGE")
        ),
        "xi_advection_successor_scope": (
            "#601 planar one-depth and #605 n16 single-ground-depth q4 remain "
            "scoped controls; the named successor is depth-stratified, "
            "object-local xi advection in PREDICT, not a blanket family claim"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "research_only": True,
        "execution_allowed_by_this_receipt": False,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_review_required": True,
    }
    _write_json(final_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("validate", "preflight", "run"))
    args = parser.parse_args(argv)
    config = PC2PoseDescentConfigV1.from_path(args.config)
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "typed_config_hash": config.typed_hash(),
                    "bindings": config.validate_all_bindings(),
                    "score_domain_loss": config.score_domain_loss,
                    "pose_marginal_weight_law": (config.pose_marginal_weight_law),
                    "pose_objective_weight": config.pose_objective_weight,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "preflight":
        receipt = _preflight(config)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    receipt = _run(config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "verdict_scope": receipt["verdict_scope"],
                "accepted_steps": receipt["governor"]["accepted_steps"],
                "slope_rows": receipt["slope_rows"],
                "main_review_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PC2PoseDescentError as exc:
        raise SystemExit(str(exc)) from exc
