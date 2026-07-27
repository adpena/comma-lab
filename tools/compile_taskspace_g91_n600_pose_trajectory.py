#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and measure fresh G91 n600 XIP2 trajectories on the exact G88 seam.

This runner is intentionally resumable.  Every scorer batch is an atomic
checkpoint, every treatment has a distinct stage directory, and the exact
archive/XIP2/member bytes are written before scorer work begins.  It streams at
most ``batch_pairs <= 16`` from the retained G85 deterministic double decode and
never materializes another multi-gigabyte raw output.

The default ``all`` phase performs:

1. SSD storage/custody preflight and source-video PoseNet target self-check;
2. fresh trajectory derivation, factorability analysis, XIP2 coder/ZIP race;
3. bounded operational timing smoke (explicitly not decision evidence);
4. full n600 frozen CPU-PoseNet PASS and XIP2 measurements;
5. exact sparse PASS/XIP2 allocation selection and selected double decode.

This is local CPU research signal, never an exact contest score.  Public
``inflate.sh`` runtime linkage and ``upstream/evaluate.py`` remain separate
promotion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from comma_lab.operator_storage_waterfall import (  # noqa: E402
    operator_storage_policy_payload,
    operator_work_tiers,
)
from tac.witness_dsl import taskspace_g88_population_conditional_y0_pvsa_v1 as g88  # noqa: E402
from tac.witness_dsl.taskspace_g91_n600_pose_trajectory_compiler_v1 import (  # noqa: E402
    AUTHORITY_BLOCKER,
    CODER_UNIVERSE,
    COMPILER_ID,
    DIRECT_POSE_TARGET_AS_WARP_CONTROL_ADMISSIBLE,
    FRESH_TARGET_CUSTODY_QUALIFIER,
    G16_SETTLED_NEGATIVE_AUTHORITY,
    INVERSE_CONTROL_SOLVED,
    ONLY_ADMISSIBLE_PROMOTION_PATH,
    PAIR_COUNT,
    RATE_DENOMINATOR_BYTES,
    SELECTOR_POLICY_ID,
    TRAJECTORY_POLICY_ID,
    G91ModeSelectionRowV1,
    G91TrajectoryTreatmentV1,
    G91TrajectoryV1,
    compile_coder_race,
    compile_measured_mode_selection,
    derive_fresh_trajectory,
    select_measured_mode_or_base,
)

SCHEMA = "tac.g91_n600_pose_trajectory_runner_config.v1"
PREFLIGHT_SCHEMA = "tac.g91_n600_pose_trajectory_storage_custody_preflight.v1"
RESULT_SCHEMA = "tac.g91_n600_pose_trajectory_measurement.v1"
PAIR_SHAPE = (2, 874, 1164, 3)
RAW_SHAPE = (PAIR_COUNT, *PAIR_SHAPE)
RAW_BYTES = int(np.prod(RAW_SHAPE, dtype=np.int64))
MAX_MEMBER_BYTES = 2 << 20
MAX_SECTION_BYTES = 1 << 20
SELECTION_STAGE = "stage_05_selection_exhaustive_full_prefix_v3"


class G91RunnerError(RuntimeError):
    """A G91 launch, custody, checkpoint, or scorer contract failed."""


def _sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise G91RunnerError(f"immutable artifact differs on resume: {path}")
        return
    _atomic_write(path, payload)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    _write_once_or_equal(path, _canonical_json(payload))


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_json(payload))


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise G91RunnerError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise G91RunnerError(f"{label} must be one JSON object")
    return value


def _exact_path_identity(spec: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(str(spec["path"])).resolve(strict=True)
        expected_bytes = int(spec["bytes"])
        expected_sha = str(spec["sha256"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise G91RunnerError(f"{label} identity is malformed") from exc
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise G91RunnerError(f"{label} bytes differ from sealed custody")
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise G91RunnerError(f"{label} SHA-256 differs from sealed custody")
    return path


def _fp32(value: object, label: str) -> float:
    try:
        result = float(np.float32(float(value)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise G91RunnerError(f"{label} must fit finite fp32") from exc
    if not math.isfinite(result):
        raise G91RunnerError(f"{label} must fit finite fp32")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_mapping(path, "G91 config")
    if config.get("schema") != SCHEMA:
        raise G91RunnerError(f"config schema must be {SCHEMA}")
    if config.get("research_only") is not True or config.get("score_claim") is not False:
        raise G91RunnerError("config truth labels must remain research_only/score_claim=false")
    if int(config.get("seed", -1)) < 0:
        raise G91RunnerError("seed must be a nonnegative exact integer")
    if not 1 <= int(config.get("batch_pairs", 0)) <= 16:
        raise G91RunnerError("batch_pairs must be in [1,16]")
    if not 1 <= int(config.get("smoke_pairs", 0)) <= 16:
        raise G91RunnerError("smoke_pairs must be in [1,16]")
    treatments = config.get("treatments")
    if not isinstance(treatments, list) or not treatments:
        raise G91RunnerError("config requires at least one typed treatment")
    ids: set[str] = set()
    for index, row in enumerate(treatments):
        if not isinstance(row, dict):
            raise G91RunnerError(f"treatments[{index}] must be one mapping")
        treatment_id = str(row.get("treatment_id", ""))
        if treatment_id in ids:
            raise G91RunnerError("treatment IDs must be unique")
        ids.add(treatment_id)
        G91TrajectoryTreatmentV1(
            treatment_id=treatment_id,
            s_t=_fp32(row.get("s_t"), f"{treatment_id}.s_t"),
            s_r=_fp32(row.get("s_r"), f"{treatment_id}.s_r"),
            pitch=_fp32(row.get("pitch"), f"{treatment_id}.pitch"),
            centered_rank=int(row.get("centered_rank")),
            q_levels=int(row.get("q_levels")),
        )
    full_ids = config.get("full_treatment_ids")
    if (
        not isinstance(full_ids, list)
        or not full_ids
        or any(type(value) is not str or value not in ids for value in full_ids)
        or len(set(full_ids)) != len(full_ids)
    ):
        raise G91RunnerError("full_treatment_ids must be a nonempty unique treatment subset")
    coders = config.get("coders")
    if (
        not isinstance(coders, list)
        or not coders
        or len(set(coders)) != len(coders)
        or any(value not in CODER_UNIVERSE for value in coders)
    ):
        raise G91RunnerError("coders escaped the closed G91/XIP2 universe")
    return config


def _treatments(config: Mapping[str, Any]) -> dict[str, G91TrajectoryTreatmentV1]:
    result: dict[str, G91TrajectoryTreatmentV1] = {}
    for row in config["treatments"]:
        treatment = G91TrajectoryTreatmentV1(
            treatment_id=str(row["treatment_id"]),
            s_t=_fp32(row["s_t"], "s_t"),
            s_r=_fp32(row["s_r"], "s_r"),
            pitch=_fp32(row["pitch"], "pitch"),
            centered_rank=int(row["centered_rank"]),
            q_levels=int(row["q_levels"]),
        )
        result[treatment.treatment_id] = treatment
    return result


def _configure_determinism(config: Mapping[str, Any]) -> None:
    import torch

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def _storage_and_custody_preflight(
    config_path: Path,
    config: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    tiers = operator_work_tiers()
    resolved_root = run_root.resolve(strict=False)
    selected_tier = None
    for tier in tiers:
        tier_root = Path(tier.root).resolve(strict=False)
        try:
            resolved_root.relative_to(tier_root)
        except ValueError:
            continue
        selected_tier = tier
        break
    if selected_tier is None or selected_tier.priority != 0:
        raise G91RunnerError("G91 output_root must be under the first operator SSD tier")
    run_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_root)
    reserve = int(config["safety_reserve_bytes"])
    if usage.free <= reserve:
        raise G91RunnerError("SSD storage preflight failed closed below safety reserve")
    identities: dict[str, dict[str, Any]] = {}
    for key in (
        "source_video",
        "base_archive",
        "base_member",
        "g85_decode_a",
        "g85_decode_b",
        "gt_cache",
        "posenet_weights",
    ):
        spec = config[key]
        path = _exact_path_identity(spec, key)
        identities[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": str(spec["sha256"]),
        }
    for key in ("g85_decode_a", "g85_decode_b"):
        if identities[key]["bytes"] != RAW_BYTES:
            raise G91RunnerError(f"{key} changed exact n600 camera ABI")
    if identities["g85_decode_a"]["sha256"] != identities["g85_decode_b"]["sha256"]:
        raise G91RunnerError("G85 retained double decode hashes differ")
    if identities["base_member"]["sha256"] != config["base_member_sha256"]:
        raise G91RunnerError("base member foreign key differs from config")
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "config": {
            "path": str(config_path.resolve()),
            "bytes": config_path.stat().st_size,
            "sha256": _sha256_file(config_path),
        },
        "operator_storage_policy": operator_storage_policy_payload(),
        "selected_tier": {
            "name": selected_tier.name,
            "root": selected_tier.root,
            "priority": selected_tier.priority,
        },
        "output_root": str(run_root),
        "free_bytes_at_preflight": usage.free,
        "safety_reserve_bytes": reserve,
        "storage_status": "PASS",
        "large_artifact_policy": {
            "new_raw_outputs_materialized": False,
            "stream_batch_pairs_maximum": int(config["batch_pairs"]),
            "automatic_cleanup": (
                "atomic temporary files are success-cleaned; no rebuildable bulky output "
                "is created; retained G85 raws remain immutable inputs"
            ),
            "certify_or_block": True,
        },
        "identities": identities,
        "false_authority": {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "public_runtime_closed": False,
            "exact_contest_eval": False,
        },
    }
    preflight_path = run_root / "stage_00_preflight/preflight.json"
    if preflight_path.exists():
        preserved = _read_mapping(preflight_path, "preserved G91 preflight")
        payload["free_bytes_at_preflight"] = preserved.get("free_bytes_at_preflight")
    _write_json_once(preflight_path, payload)
    return payload


def _load_gt_poses(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        poses = np.asarray(archive["gt_poses"])
    if poses.dtype != np.float64 or poses.shape != (PAIR_COUNT, 6) or not np.all(np.isfinite(poses)):
        raise G91RunnerError("GT cache pose member changed exact float64 [600,6] ABI")
    return np.ascontiguousarray(poses)


def _load_posenet(weights: Path):
    import torch
    from modules import PoseNet
    from safetensors.torch import load_file

    model = PoseNet().eval()
    model.load_state_dict(load_file(str(weights), device="cpu"))
    model = model.to(torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _pose_predictions_batch(posenet, pairs_uint8: np.ndarray) -> np.ndarray:
    import einops
    import torch

    pairs = np.asarray(pairs_uint8)
    if pairs.dtype != np.uint8 or pairs.ndim != 5 or pairs.shape[1:] != PAIR_SHAPE:
        raise G91RunnerError("PoseNet batch changed exact uint8 [N,2,874,1164,3] ABI")
    # G88 result arrays are intentionally immutable.  Copy before the Torch
    # view so its writable-buffer precondition is explicit and warning-free.
    tensor = torch.from_numpy(np.ascontiguousarray(pairs).copy()).float()
    x = einops.rearrange(tensor, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        output = posenet(posenet.preprocess_input(x))
        pose = output["pose"] if isinstance(output, dict) else output
        half = next(
            (int(head.out // 2) for head in posenet.hydra.heads if head.name == "pose"),
            int(pose.shape[-1] // 2),
        )
        result = pose[:, :half].cpu().numpy().astype(np.float64)
    if result.shape != (pairs.shape[0], 6):
        raise G91RunnerError("PoseNet output changed exact first-six target ABI")
    return result


def _pose_losses_batch(
    posenet,
    pairs_uint8: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = _pose_predictions_batch(posenet, pairs_uint8)
    target = np.asarray(targets, dtype=np.float64)
    if target.shape != prediction.shape:
        raise G91RunnerError("PoseNet targets differ from batch prediction shape")
    losses = np.mean((prediction - target) ** 2, axis=1).astype(np.float64)
    return losses, prediction


def _source_pose_selfcheck(
    *,
    source_video: Path,
    gt_poses: np.ndarray,
    posenet,
    pair_count: int,
) -> dict[str, Any]:
    import torch
    from frame_utils import AVVideoDataset

    dataset = AVVideoDataset(
        [source_video.name],
        data_dir=source_video.parent,
        batch_size=pair_count,
        device=torch.device("cpu"),
        num_threads=1,
        seed=1234,
        prefetch_queue_depth=1,
    )
    dataset.prepare_data()
    _, _, pairs = next(iter(dataset))
    batch = np.ascontiguousarray(pairs[:pair_count].cpu().numpy(), dtype=np.uint8)
    losses, predictions = _pose_losses_batch(
        posenet,
        batch,
        gt_poses[:pair_count],
    )
    maximum = float(np.max(losses))
    if maximum >= 1e-6:
        raise G91RunnerError(f"source-video PoseNet target self-check failed: maximum MSE {maximum:.9g}")
    return {
        "pair_count": pair_count,
        "source_pair_sha256": _sha256_array(batch),
        "target_sha256": _sha256_array(gt_poses[:pair_count]),
        "prediction_sha256": _sha256_array(predictions),
        "maximum_pair_mse": maximum,
        "status": "PASS",
    }


def _compile_stage(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    poses: np.ndarray,
    base_member: bytes,
) -> tuple[
    dict[str, G91TrajectoryV1],
    dict[str, Any],
]:
    trajectories: dict[str, G91TrajectoryV1] = {}
    treatment_receipts: list[dict[str, Any]] = []
    coders = tuple(str(value) for value in config["coders"])
    for treatment_id, treatment in _treatments(config).items():
        stage = run_root / f"stage_01_compile/{treatment_id}"
        trajectory = derive_fresh_trajectory(poses, treatment)
        race = compile_coder_race(
            trajectory=trajectory,
            base_pvsa_member_bytes=base_member,
            semantic_p_sha256=str(config["semantic_p_sha256"]),
            default_mode="XIP2_SE3_FRAME0_WARP",
            xip2_pair_ids=tuple(range(PAIR_COUNT)),
            coders=coders,
            maximum_member_bytes=MAX_MEMBER_BYTES,
            maximum_section_bytes=MAX_SECTION_BYTES,
        )
        selected = race[0]
        _write_once_or_equal(stage / "selected.xip2", selected.xip2_payload)
        _write_once_or_equal(
            stage / "selected.operand",
            selected.parsed_operand.to_bytes(),
        )
        _write_once_or_equal(
            stage / "selected.0.bin",
            selected.archive_build.selected.member_bytes,
        )
        _write_once_or_equal(
            stage / "selected.archive.zip",
            selected.archive_build.outer_build.selected.archive_bytes,
        )
        receipt = {
            "schema": "tac.g91_trajectory_compile_stage.v1",
            "treatment": {
                "treatment_id": treatment.treatment_id,
                "s_t": treatment.s_t,
                "s_r": treatment.s_r,
                "pitch": treatment.pitch,
                "centered_rank": treatment.centered_rank,
                "q_levels": treatment.q_levels,
            },
            "source_target_sha256": trajectory.source_target_sha256,
            "calibrated_xi_sha256": _sha256_array(trajectory.calibrated_xi),
            "factorized_xi_sha256": _sha256_array(trajectory.factorized_xi),
            "decoded_xi_sha256": _sha256_array(trajectory.decoded_xi),
            "factorability": trajectory.factorability,
            "coder_race": [row.receipt() for row in race],
            "selected_coder": selected.coder,
            "selected_archive_path": str(stage / "selected.archive.zip"),
            "selected_member_path": str(stage / "selected.0.bin"),
            "selected_xip2_path": str(stage / "selected.xip2"),
            "selected_operand_path": str(stage / "selected.operand"),
            "strict_xip2_parseback_equal": True,
            "strict_outer_store_deflate_parseback_equal": True,
            "checkpoint_kind": "BYTE_CLOSE_LOADABLE_END_OF_TREATMENT_STAGE",
            "research_only": True,
            "score_claim": False,
        }
        _write_json_once(stage / "receipt.json", receipt)
        trajectories[treatment_id] = trajectory
        treatment_receipts.append(receipt)
    aggregate = {
        "schema": "tac.g91_trajectory_compile_aggregate.v1",
        "compiler_id": COMPILER_ID,
        "trajectory_policy_id": TRAJECTORY_POLICY_ID,
        "selector_policy_id": SELECTOR_POLICY_ID,
        "base_member_sha256": _sha256_bytes(base_member),
        "semantic_p_sha256": config["semantic_p_sha256"],
        "source_target_sha256": _sha256_array(poses),
        "treatments": treatment_receipts,
        "fresh_payload_rule": (
            "all XIP2 bytes derived in this run from current source GT target custody; "
            "no historical/public XIP2/V15/C1 payload accepted"
        ),
        "research_only": True,
        "score_claim": False,
    }
    _write_json_once(run_root / "stage_01_compile/aggregate.json", aggregate)
    return trajectories, aggregate


def _batch_ranges(pair_count: int, batch_pairs: int) -> Iterable[tuple[int, int]]:
    for start in range(0, pair_count, batch_pairs):
        yield start, min(start + batch_pairs, pair_count)


def _load_progress(path: Path, treatment_id: str, pair_count: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "tac.g91_pose_measurement_progress.v1",
            "treatment_id": treatment_id,
            "pair_count": pair_count,
            "completed_stop": 0,
            "batches": [],
            "d_pose": [],
            "prediction_rows": [],
        }
    progress = _read_mapping(path, "pose progress")
    if (
        progress.get("schema") != "tac.g91_pose_measurement_progress.v1"
        or progress.get("treatment_id") != treatment_id
        or int(progress.get("pair_count", -1)) != pair_count
        or int(progress.get("completed_stop", -1)) != len(progress.get("d_pose", []))
    ):
        raise G91RunnerError("pose progress checkpoint custody differs")
    return progress


def _score_pass_population(
    *,
    run_root: Path,
    pair_count: int,
    batch_pairs: int,
    raw_a: np.memmap,
    raw_b: np.memmap,
    gt_poses: np.ndarray,
    posenet,
    stage_name: str,
) -> dict[str, Any]:
    stage = run_root / stage_name / "pass"
    progress_path = stage / "progress.json"
    progress = _load_progress(progress_path, "PASS_P0", pair_count)
    start_at = int(progress["completed_stop"])
    started = time.monotonic()
    for start, stop in _batch_ranges(pair_count, batch_pairs):
        if stop <= start_at:
            continue
        if start != start_at:
            raise G91RunnerError("PASS resume checkpoint is not batch-contiguous")
        first = np.asarray(raw_a[start:stop]).copy()
        second = np.asarray(raw_b[start:stop]).copy()
        if not np.array_equal(first, second):
            raise G91RunnerError("G85 deterministic double decode differs in PASS batch")
        losses, prediction = _pose_losses_batch(
            posenet,
            first,
            gt_poses[start:stop],
        )
        progress["batches"].append(
            {
                "pair_start": start,
                "pair_stop": stop,
                "camera_sha256": _sha256_array(first),
                "exact_y1_sha256": _sha256_array(first[:, 1]),
                "mean_d_pose": float(np.mean(losses)),
            }
        )
        progress["d_pose"].extend(losses.tolist())
        progress["prediction_rows"].extend(prediction.tolist())
        progress["completed_stop"] = stop
        _write_progress(progress_path, progress)
        start_at = stop
        print(
            f"[G91] {stage_name} PASS {stop}/{pair_count} mean={np.mean(progress['d_pose']):.9g}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    losses = np.asarray(progress["d_pose"], dtype=np.float64)
    predictions = np.asarray(progress["prediction_rows"], dtype=np.float64)
    if losses.shape != (pair_count,) or predictions.shape != (pair_count, 6):
        raise G91RunnerError("PASS completed checkpoint changed exact population shape")
    final = {
        "schema": "tac.g91_pose_measurement_stage.v1",
        "treatment_id": "PASS_P0",
        "pair_count": pair_count,
        "d_pose_mean": float(np.mean(losses)),
        "d_pose_sha256": _sha256_array(losses),
        "prediction_sha256": _sha256_array(predictions),
        "ordered_batch_camera_digest": _sha256_bytes(
            _canonical_json({"batch_camera_sha256": [row["camera_sha256"] for row in progress["batches"]]})
        ),
        "batch_count": len(progress["batches"]),
        "elapsed_seconds_this_invocation": elapsed,
        "checkpoint_kind": "COMPLETE_PER_PAIR_FROZEN_POSE_LOSSES_AND_PREDICTIONS",
        "research_only": True,
        "score_claim": False,
    }
    final_path = stage / "final.json"
    if final_path.exists():
        preserved = _read_mapping(final_path, "preserved PASS final checkpoint")
        final["elapsed_seconds_this_invocation"] = preserved.get("elapsed_seconds_this_invocation")
    _write_json_once(final_path, final)
    return {**final, "d_pose": losses, "predictions": predictions}


def _score_xip2_population(
    *,
    run_root: Path,
    pair_count: int,
    batch_pairs: int,
    raw_a: np.memmap,
    raw_b: np.memmap,
    gt_poses: np.ndarray,
    posenet,
    treatment_id: str,
    operand: g88.PopulationConditionalOperandV1,
    stage_name: str,
) -> dict[str, Any]:
    stage = run_root / stage_name / treatment_id
    progress_path = stage / "progress.json"
    progress = _load_progress(progress_path, treatment_id, pair_count)
    start_at = int(progress["completed_stop"])
    started = time.monotonic()
    for start, stop in _batch_ranges(pair_count, batch_pairs):
        if stop <= start_at:
            continue
        if start != start_at:
            raise G91RunnerError("XIP2 resume checkpoint is not batch-contiguous")
        first = np.asarray(raw_a[start:stop]).copy()
        second = np.asarray(raw_b[start:stop]).copy()
        result = g88.apply_population_conditional_to_decoded_batch(
            operand=operand,
            first_base_camera_pairs=first,
            second_base_camera_pairs=second,
            local_pair_ids=tuple(range(start, stop)),
        )
        if not result.deterministic_double_decode or not np.array_equal(
            result.camera_pairs[:, 1],
            first[:, 1],
        ):
            raise G91RunnerError("G88 XIP2 batch lost double-decode or exact-Y1 proof")
        losses, prediction = _pose_losses_batch(
            posenet,
            result.camera_pairs,
            gt_poses[start:stop],
        )
        progress["batches"].append(
            {
                "pair_start": start,
                "pair_stop": stop,
                "camera_sha256": result.camera_sha256,
                "base_camera_sha256": result.base_camera_sha256,
                "exact_y1_sha256": result.exact_y1_sha256,
                "owned_y0_sha256": result.owned_y0_sha256,
                "changed_y0_values": result.changed_y0_values,
                "changed_y0_pixels": result.changed_y0_pixels,
                "active_pair_ids": list(result.active_pair_ids),
                "mean_d_pose": float(np.mean(losses)),
            }
        )
        progress["d_pose"].extend(losses.tolist())
        progress["prediction_rows"].extend(prediction.tolist())
        progress["completed_stop"] = stop
        _write_progress(progress_path, progress)
        start_at = stop
        print(
            f"[G91] {stage_name} {treatment_id} {stop}/{pair_count} mean={np.mean(progress['d_pose']):.9g}",
            flush=True,
        )
    elapsed = time.monotonic() - started
    losses = np.asarray(progress["d_pose"], dtype=np.float64)
    predictions = np.asarray(progress["prediction_rows"], dtype=np.float64)
    if losses.shape != (pair_count,) or predictions.shape != (pair_count, 6):
        raise G91RunnerError("XIP2 completed checkpoint changed exact population shape")
    batch_rows = progress["batches"]
    final = {
        "schema": "tac.g91_pose_measurement_stage.v1",
        "treatment_id": treatment_id,
        "pair_count": pair_count,
        "d_pose_mean": float(np.mean(losses)),
        "d_pose_sha256": _sha256_array(losses),
        "prediction_sha256": _sha256_array(predictions),
        "ordered_batch_camera_digest": _sha256_bytes(
            _canonical_json({"batch_camera_sha256": [row["camera_sha256"] for row in batch_rows]})
        ),
        "ordered_batch_y1_digest": _sha256_bytes(
            _canonical_json({"batch_y1_sha256": [row["exact_y1_sha256"] for row in batch_rows]})
        ),
        "changed_y0_values": int(sum(int(row["changed_y0_values"]) for row in batch_rows)),
        "changed_y0_pixels": int(sum(int(row["changed_y0_pixels"]) for row in batch_rows)),
        "deterministic_double_decode_every_batch": True,
        "exact_y1_preserved_every_batch": True,
        "batch_count": len(batch_rows),
        "elapsed_seconds_this_invocation": elapsed,
        "checkpoint_kind": "COMPLETE_PER_PAIR_FROZEN_POSE_LOSSES_AND_PREDICTIONS",
        "research_only": True,
        "score_claim": False,
    }
    final_path = stage / "final.json"
    if final_path.exists():
        preserved = _read_mapping(final_path, "preserved XIP2 final checkpoint")
        final["elapsed_seconds_this_invocation"] = preserved.get("elapsed_seconds_this_invocation")
    _write_json_once(final_path, final)
    return {**final, "d_pose": losses, "predictions": predictions}


def _global_operand_for_trajectory(
    *,
    trajectory: G91TrajectoryV1,
    base_member: bytes,
    config: Mapping[str, Any],
):
    race = compile_coder_race(
        trajectory=trajectory,
        base_pvsa_member_bytes=base_member,
        semantic_p_sha256=str(config["semantic_p_sha256"]),
        default_mode="XIP2_SE3_FRAME0_WARP",
        xip2_pair_ids=tuple(range(PAIR_COUNT)),
        coders=tuple(str(value) for value in config["coders"]),
        maximum_member_bytes=MAX_MEMBER_BYTES,
        maximum_section_bytes=MAX_SECTION_BYTES,
    )
    return race[0]


def _run_smoke(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    trajectories: Mapping[str, G91TrajectoryV1],
    base_member: bytes,
    raw_a: np.memmap,
    raw_b: np.memmap,
    gt_poses: np.ndarray,
    posenet,
) -> dict[str, Any]:
    pair_count = int(config["smoke_pairs"])
    pass_result = _score_pass_population(
        run_root=run_root,
        pair_count=pair_count,
        batch_pairs=min(int(config["batch_pairs"]), pair_count),
        raw_a=raw_a,
        raw_b=raw_b,
        gt_poses=gt_poses,
        posenet=posenet,
        stage_name="stage_02_operational_smoke",
    )
    rows = []
    for treatment_id, trajectory in trajectories.items():
        selected = _global_operand_for_trajectory(
            trajectory=trajectory,
            base_member=base_member,
            config=config,
        )
        result = _score_xip2_population(
            run_root=run_root,
            pair_count=pair_count,
            batch_pairs=min(int(config["batch_pairs"]), pair_count),
            raw_a=raw_a,
            raw_b=raw_b,
            gt_poses=gt_poses,
            posenet=posenet,
            treatment_id=treatment_id,
            operand=selected.parsed_operand,
            stage_name="stage_02_operational_smoke",
        )
        rows.append(
            {
                "treatment_id": treatment_id,
                "d_pose_mean": result["d_pose_mean"],
                "delta_vs_pass": result["d_pose_mean"] - pass_result["d_pose_mean"],
                "selected_outer_bytes": selected.selected_outer_bytes,
            }
        )
    payload = {
        "schema": "tac.g91_operational_timing_smoke.v1",
        "pair_count": pair_count,
        "pass_d_pose_mean": pass_result["d_pose_mean"],
        "treatments": rows,
        "evidence_status": "OPERATIONAL_TIMING_ONLY_NOT_A_FINDING_NOT_SELECTOR_INPUT",
        "full_treatment_ids_preregistered_in_config": config["full_treatment_ids"],
        "research_only": True,
        "score_claim": False,
    }
    _write_json_once(run_root / "stage_02_operational_smoke/aggregate.json", payload)
    return payload


def _materialize_selected_bytes(
    stage: Path,
    selection: G91ModeSelectionRowV1,
) -> dict[str, Any]:
    coder = selection.coder_row
    _write_once_or_equal(stage / "selected.xip2", coder.xip2_payload)
    _write_once_or_equal(
        stage / "selected.operand",
        coder.parsed_operand.to_bytes(),
    )
    _write_once_or_equal(
        stage / "selected.0.bin",
        coder.archive_build.selected.member_bytes,
    )
    _write_once_or_equal(
        stage / "selected.archive.zip",
        coder.archive_build.outer_build.selected.archive_bytes,
    )
    return {
        "xip2": {
            "path": str(stage / "selected.xip2"),
            "bytes": len(coder.xip2_payload),
            "sha256": coder.xip2_sha256,
        },
        "operand": {
            "path": str(stage / "selected.operand"),
            "bytes": coder.operand_bytes,
            "sha256": coder.operand_sha256,
        },
        "member": {
            "path": str(stage / "selected.0.bin"),
            "bytes": coder.successor_member_bytes,
            "sha256": coder.successor_member_sha256,
        },
        "archive": {
            "path": str(stage / "selected.archive.zip"),
            "bytes": coder.selected_outer_bytes,
            "sha256": coder.selected_outer_sha256,
            "encoding": coder.selected_outer_encoding,
        },
    }


def _verify_selected_double_decode(
    *,
    selection: G91ModeSelectionRowV1,
    raw_a: np.memmap,
    raw_b: np.memmap,
    batch_pairs: int,
) -> dict[str, Any]:
    camera_hashes: list[str] = []
    y1_hashes: list[str] = []
    changed_values = 0
    changed_pixels = 0
    for start, stop in _batch_ranges(PAIR_COUNT, batch_pairs):
        first = np.asarray(raw_a[start:stop]).copy()
        second = np.asarray(raw_b[start:stop]).copy()
        result = g88.apply_population_conditional_to_decoded_batch(
            operand=selection.coder_row.parsed_operand,
            first_base_camera_pairs=first,
            second_base_camera_pairs=second,
            local_pair_ids=tuple(range(start, stop)),
        )
        if not result.deterministic_double_decode or not np.array_equal(
            result.camera_pairs[:, 1],
            first[:, 1],
        ):
            raise G91RunnerError("selected sparse allocation failed exact double decode")
        expected_active = tuple(pair_id for pair_id in range(start, stop) if pair_id in set(selection.xip2_pair_ids))
        if result.active_pair_ids != expected_active:
            raise G91RunnerError("selected sparse allocation active IDs differ on decode")
        camera_hashes.append(result.camera_sha256)
        y1_hashes.append(result.exact_y1_sha256)
        changed_values += result.changed_y0_values
        changed_pixels += result.changed_y0_pixels
    return {
        "batch_pairs_maximum": batch_pairs,
        "batch_count": len(camera_hashes),
        "deterministic_double_decode_every_batch": True,
        "exact_y1_preserved_every_batch": True,
        "ordered_batch_camera_digest": _sha256_bytes(_canonical_json({"batch_camera_sha256": camera_hashes})),
        "ordered_batch_y1_digest": _sha256_bytes(_canonical_json({"batch_y1_sha256": y1_hashes})),
        "changed_y0_values": changed_values,
        "changed_y0_pixels": changed_pixels,
    }


def _run_full_and_select(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    trajectories: Mapping[str, G91TrajectoryV1],
    base_member: bytes,
    raw_a: np.memmap,
    raw_b: np.memmap,
    gt_poses: np.ndarray,
    posenet,
) -> dict[str, Any]:
    batch_pairs = int(config["batch_pairs"])
    pass_result = _score_pass_population(
        run_root=run_root,
        pair_count=PAIR_COUNT,
        batch_pairs=batch_pairs,
        raw_a=raw_a,
        raw_b=raw_b,
        gt_poses=gt_poses,
        posenet=posenet,
        stage_name="stage_03_full_n600",
    )
    base_archive_bytes = int(config["base_archive"]["bytes"])
    base_pose_term = math.sqrt(10.0 * pass_result["d_pose_mean"])
    base_rate_term = 25.0 * base_archive_bytes / RATE_DENOMINATOR_BYTES
    base_objective = base_pose_term + base_rate_term
    treatment_rows: list[dict[str, Any]] = []
    best_selection: tuple[str, G91ModeSelectionRowV1] | None = None
    for treatment_id in config["full_treatment_ids"]:
        trajectory = trajectories[str(treatment_id)]
        global_coder = _global_operand_for_trajectory(
            trajectory=trajectory,
            base_member=base_member,
            config=config,
        )
        result = _score_xip2_population(
            run_root=run_root,
            pair_count=PAIR_COUNT,
            batch_pairs=batch_pairs,
            raw_a=raw_a,
            raw_b=raw_b,
            gt_poses=gt_poses,
            posenet=posenet,
            treatment_id=str(treatment_id),
            operand=global_coder.parsed_operand,
            stage_name="stage_03_full_n600",
        )
        selections = compile_measured_mode_selection(
            trajectory=trajectory,
            base_pvsa_member_bytes=base_member,
            semantic_p_sha256=str(config["semantic_p_sha256"]),
            pass_d_pose=pass_result["d_pose"],
            xip2_d_pose=result["d_pose"],
            coders=tuple(str(value) for value in config["coders"]),
            maximum_member_bytes=MAX_MEMBER_BYTES,
            maximum_section_bytes=MAX_SECTION_BYTES,
        )
        if not selections:
            raise G91RunnerError("measured selector produced no executable XIP2 allocation")
        selected = selections[0]
        if best_selection is None or selected.selector_objective < best_selection[1].selector_objective:
            best_selection = (str(treatment_id), selected)
        per_treatment = {
            "treatment_id": treatment_id,
            "global_xip2_d_pose_mean": result["d_pose_mean"],
            "pass_d_pose_mean": pass_result["d_pose_mean"],
            "global_xip2_delta_vs_pass": (result["d_pose_mean"] - pass_result["d_pose_mean"]),
            "global_xip2_exact_bytes": global_coder.receipt(),
            "measured_pairwise": {
                "xip2_better_pair_count": int(np.count_nonzero(result["d_pose"] < pass_result["d_pose"])),
                "pass_better_pair_count": int(np.count_nonzero(pass_result["d_pose"] < result["d_pose"])),
                "ties": int(np.count_nonzero(pass_result["d_pose"] == result["d_pose"])),
            },
            "selection_rows": [row.receipt() for row in selections],
            "selected": selected.receipt(),
            "strictly_beats_unchanged_base": selected.selector_objective < base_objective,
        }
        _write_json_once(
            run_root / f"{SELECTION_STAGE}/{treatment_id}/selection.json",
            per_treatment,
        )
        treatment_rows.append(per_treatment)
    if best_selection is None:
        raise G91RunnerError("no full n600 treatment was measured")
    best_treatment_id, best_executable = best_selection
    selected = select_measured_mode_or_base(
        rows=(best_executable,),
        pass_d_pose=pass_result["d_pose"],
        base_outer_bytes=base_archive_bytes,
    )
    selected_stage = run_root / SELECTION_STAGE / "selected"
    invariant_d_seg = float(config["g85_d_seg"])
    base_combined = 100.0 * invariant_d_seg + base_objective
    if selected is None:
        selected_treatment_id: str | None = None
        selected_bytes = {
            "decision": "PRESERVE_UNCHANGED_BASE_ARCHIVE",
            "archive": dict(config["base_archive"]),
            "member": dict(config["base_member"]),
        }
        double_decode = {
            "decision": "REUSE_SEALED_G85_BASE_DOUBLE_DECODE",
            "decode_a": dict(config["g85_decode_a"]),
            "decode_b": dict(config["g85_decode_b"]),
            "byte_identical_sha256": config["g85_decode_a"]["sha256"],
        }
        selected_pose_term = base_pose_term
        selected_rate_term = base_rate_term
        selected_combined = base_combined
        selected_receipt: dict[str, Any] | None = None
        decision = "NO_ACTIVE_G91_CANDIDATE_PRESERVE_EXACT_BASE_K0_DEFAULT_PASS"
    else:
        selected_treatment_id = best_treatment_id
        selected_bytes = _materialize_selected_bytes(selected_stage, selected)
        double_decode = _verify_selected_double_decode(
            selection=selected,
            raw_a=raw_a,
            raw_b=raw_b,
            batch_pairs=batch_pairs,
        )
        selected_pose_term = selected.pose_score_term
        selected_rate_term = selected.rate_score_term
        selected_combined = 100.0 * invariant_d_seg + selected_pose_term + selected_rate_term
        selected_receipt = selected.receipt()
        decision = "ACTIVE_G91_EXECUTABLE_PREFIX_STRICTLY_BEATS_EXACT_BASE"
    outcome = {
        "schema": RESULT_SCHEMA,
        "compiler_id": COMPILER_ID,
        "trajectory_policy_id": TRAJECTORY_POLICY_ID,
        "selector_policy_id": SELECTOR_POLICY_ID,
        "selector_sparse_count_grid": (
            "default XIP2: K=0..599 PASS exceptions; default PASS: "
            "K=1..600 XIP2 activations; both orders cover all 600 pairs"
        ),
        "selector_scope": (
            "exhaustive over every prefix length K in both stable measured-benefit orders, "
            "including prefixes beyond the positive local-benefit boundary; not a claim of "
            "combinatorial 2^600 all-subset optimality because exact DEFLATE bytes depend "
            "on the pair-ID pattern"
        ),
        "sample_count": PAIR_COUNT,
        "authority": "[macOS-CPU frozen-torch local research-signal]",
        "source_target_custody_qualifier": FRESH_TARGET_CUSTODY_QUALIFIER,
        "base": {
            "d_pose": pass_result["d_pose_mean"],
            "d_seg": invariant_d_seg,
            "archive_bytes": base_archive_bytes,
            "archive_sha256": config["base_archive"]["sha256"],
            "pose_score_term": base_pose_term,
            "rate_score_term": base_rate_term,
            "local_combined_formula_value": base_combined,
        },
        "selected_treatment_id": selected_treatment_id,
        "selection_decision": decision,
        "selected_mode_allocation": selected_receipt,
        "selected_artifacts": selected_bytes,
        "selected_double_decode": double_decode,
        "selected_pose_score_term": selected_pose_term,
        "selected_rate_score_term": selected_rate_term,
        "selected_d_seg": invariant_d_seg,
        "selected_d_seg_invariance_reason": (
            "exact decoded Y1 is byte-identical for every pair and upstream SegNet reads only the last frame"
        ),
        "selected_local_combined_formula_value": selected_combined,
        "delta_selected_minus_base": selected_combined - base_combined,
        "treatments": treatment_rows,
        "truth": {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "public_runtime_closed": False,
            "upstream_exact_n600_eval_run": False,
            "pointer_moved": False,
            "synthetic_or_inherited_xip2_used": False,
            "unchanged_base_k0_default_pass_in_decision_universe": True,
            "no_active_g91_candidate": selected is None,
            "inverse_control_solved": INVERSE_CONTROL_SOLVED,
            "direct_pose_target_as_warp_control_admissible": (DIRECT_POSE_TARGET_AS_WARP_CONTROL_ADMISSIBLE),
            "source_pose_trajectory_role": ("factorability_and_initializer_evidence_only_not_inverse_warp_control"),
            "settled_negative_authority": G16_SETTLED_NEGATIVE_AUTHORITY,
            "only_admissible_promotion_path": ONLY_ADMISSIBLE_PROMOTION_PATH,
        },
        "open_blockers": [
            AUTHORITY_BLOCKER,
            g88.PUBLIC_RUNTIME_BLOCKER,
        ],
        "pointer_delta_honesty": (
            "exact frontier pointer UNMOVED; this local frozen-PoseNet measurement "
            "and counted G88 archive are not a public upstream evaluation"
        ),
    }
    _write_json_once(selected_stage / "receipt.json", outcome)
    return outcome


def _summary_for_stdout(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": value.get("schema"),
        "selected_treatment_id": value.get("selected_treatment_id"),
        "base": value.get("base"),
        "selected_mode_allocation": value.get("selected_mode_allocation"),
        "selected_artifacts": value.get("selected_artifacts"),
        "delta_selected_minus_base": value.get("delta_selected_minus_base"),
        "open_blockers": value.get("open_blockers"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="run root containing atomic per-stage checkpoints; required even for a fresh launch",
    )
    parser.add_argument(
        "--phase",
        choices=("compile", "smoke", "full", "select", "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve(strict=True)
    config = _load_config(config_path)
    run_root = args.resume_from.resolve(strict=False)
    configured_root = Path(str(config["output_root"])).resolve(strict=False)
    if run_root != configured_root:
        raise G91RunnerError("--resume-from must equal the typed config output_root")
    if os.environ.get("TAC_GOVERNED_ADMISSION") != "1":
        raise G91RunnerError("G91 heavy entrypoint requires governed admission via tools/safe_run.py")
    _configure_determinism(config)
    preflight = _storage_and_custody_preflight(config_path, config, run_root)
    poses = _load_gt_poses(Path(preflight["identities"]["gt_cache"]["path"]))
    base_member = Path(preflight["identities"]["base_member"]["path"]).read_bytes()
    trajectories, compile_aggregate = _compile_stage(
        config=config,
        run_root=run_root,
        poses=poses,
        base_member=base_member,
    )
    if args.phase == "compile":
        print(json.dumps(compile_aggregate, sort_keys=True), flush=True)
        return 0

    posenet = _load_posenet(Path(preflight["identities"]["posenet_weights"]["path"]))
    selfcheck = _source_pose_selfcheck(
        source_video=Path(preflight["identities"]["source_video"]["path"]),
        gt_poses=poses,
        posenet=posenet,
        pair_count=int(config["selfcheck_pairs"]),
    )
    _write_json_once(run_root / "stage_00_preflight/source_pose_selfcheck.json", selfcheck)
    raw_a = np.memmap(
        preflight["identities"]["g85_decode_a"]["path"],
        dtype=np.uint8,
        mode="r",
        shape=RAW_SHAPE,
    )
    raw_b = np.memmap(
        preflight["identities"]["g85_decode_b"]["path"],
        dtype=np.uint8,
        mode="r",
        shape=RAW_SHAPE,
    )
    if args.phase in {"smoke", "all"}:
        smoke = _run_smoke(
            config=config,
            run_root=run_root,
            trajectories=trajectories,
            base_member=base_member,
            raw_a=raw_a,
            raw_b=raw_b,
            gt_poses=poses,
            posenet=posenet,
        )
        if args.phase == "smoke":
            print(json.dumps(smoke, sort_keys=True), flush=True)
            return 0
    result = _run_full_and_select(
        config=config,
        run_root=run_root,
        trajectories=trajectories,
        base_member=base_member,
        raw_a=raw_a,
        raw_b=raw_b,
        gt_poses=poses,
        posenet=posenet,
    )
    print(json.dumps(_summary_for_stdout(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
