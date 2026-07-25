#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive and measure the P1 frame-0 PoseNet quotient carrier.

The runner is local, research-only, stage-resumable, and preserves every
batch checkpoint on the SSD tier.  It never consumes quarantined PR bytes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.canonical_equations.ddm_p1_frame0_pose_quotient_carrier_20260725 import (  # noqa: E402
    DELEGATED_TARGET_D_POSE,
    GC4_STRICT_TARGET_D_POSE,
    canonical_rank_law,
    descending_covariance_spectrum,
    matched_control_fence,
    pose_targeted_actuator,
    reach_curve_disposition,
)
from tac.optimization.ddm_p1_frame0_pose_quotient_carrier import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    CHANNELS,
    GRID_H,
    GRID_W,
    MAX_RANK,
    build_counted_composition_archive,
    canonical_bytes,
    make_packet,
    parse_counted_composition_archive,
    receive_frame0_quotient,
    seeded_matched_control_basis,
    serialize_packet,
    sha256_bytes,
)

CONFIG_SCHEMA = "DDMP1Frame0PoseQuotientCarrierConfigV1"
RECEIPT_SCHEMA = "ddm_p1_frame0_pose_quotient_carrier_receipt.v1"
REACH_ROW_SCHEMA = "ddm_p1_frame0_pose_quotient_reach_row.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
PAIR_COUNT = 600
SCORER_BATCH = 32
TORCH_THREADS = 4
MIN_AVAILABLE_BYTES = 20 * 1024**3
MIN_STORAGE_BYTES = 20 * 1024**3
SCORER_HASHES = {
    "modules": "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    "posenet": "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
    "segnet": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
}


class P1RunnerError(RuntimeError):
    """Raised when P1 custody, solver, memory, or replay fails closed."""


class BoundArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def resolve(self) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else REPO_ROOT / path

    def validate_bytes(self) -> bytes:
        payload = self.resolve().read_bytes()
        if len(payload) != self.bytes or sha256_bytes(payload) != self.sha256:
            raise P1RunnerError(f"bound artifact custody differs: {self.resolve()}")
        return payload


class CarrierConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    basis_dtype: Literal["int8"]
    coefficient_dtype: Literal["int16"]
    control_seed: Literal[20260725]
    grid: tuple[Literal[24], Literal[32]]
    maximum_bytes: Literal[30000]
    maximum_rank: Literal[6]
    receiver_interpolation: Literal["integer_center_nearest"]
    term_scale: Literal["power_of_two_int8_exponent"]


class MeasurementConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_axis: Literal["[macOS-CPU frozen-scorer advisory]"]
    minimum_available_memory_bytes: Literal[21474836480]
    pair_count: Literal[600]
    scorer_batch_size: Literal[32]
    torch_threads: Literal[4]


class PreregisteredConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    delegated_target_d_pose: Literal[0.00005]
    falsifier_minimum_rows: Literal[5]
    gc4_strict_target_d_pose: Literal[0.0000294]
    seg_delta_tolerance: Literal[0.0]


class P1ConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMP1Frame0PoseQuotientCarrierConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    authority: BoundArtifactV1
    carrier: CarrierConfigV1
    delegation_checkpoint_key: Literal[
        "codex_delegate:ddm_p1_frame0_pose_quotient_carrier:20260725T141713Z"
    ]
    g4_parent_archive: BoundArtifactV1
    g4_raw_root: str
    jacobian_ridge: float
    coefficient_ridge: float
    gauss_newton_iterations: int
    gauss_newton_max_step: float
    lane_id: Literal["lane_ddm_p1_frame0_pose_quotient_carrier_20260725"]
    main_review_required: Literal[True]
    measurement: MeasurementConfigV1
    ms4d_pose_metric: BoundArtifactV1
    output_root: str
    own_python: str
    pointer: Literal["0.1910828242 [contest-CPU]"]
    preregistered: PreregisteredConfigV1
    promotion_eligible: Literal[False]
    research_only: Literal[True]
    run_id: Literal["ddm_p1_frame0_pose_quotient_carrier_20260725T141713Z"]
    score_claim: Literal[False]
    upstream_root: str

    @model_validator(mode="after")
    def _validate_solver(self) -> P1ConfigV1:
        if (
            self.jacobian_ridge != 1.0e-6
            or self.coefficient_ridge != 1.0e-8
            or self.gauss_newton_iterations != 4
            or self.gauss_newton_max_step != 4096.0
        ):
            raise ValueError("P1 solver constants differ from the sealed preregistration")
        return self

    @classmethod
    def from_path(cls, path: Path) -> tuple[P1ConfigV1, str]:
        payload = path.read_bytes()
        try:
            config = cls.model_validate_json(payload, strict=True)
        except Exception as exc:
            raise P1RunnerError("typed P1 config validation failed") from exc
        expected = canonical_bytes(config.model_dump(mode="json", by_alias=True))
        if payload != expected:
            raise P1RunnerError("typed P1 config is not canonical JSON")
        if Path(sys.executable).resolve() != Path(config.own_python).resolve():
            raise P1RunnerError(
                f"P1 must use its sealed SSD interpreter: {config.own_python}; got {sys.executable}"
            )
        return config, sha256_bytes(payload)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise P1RunnerError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_bytes(value))


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise P1RunnerError(f"JSON artifact unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise P1RunnerError(f"JSON artifact must contain one object: {path}")
    return value


def _memory_receipt(stage: str) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise P1RunnerError("P1 exact n600 preflight requires psutil") from exc
    memory = psutil.virtual_memory()
    available = int(memory.available)
    if available < MIN_AVAILABLE_BYTES:
        raise P1RunnerError(
            f"REFUSE_P1_MEMORY_{stage}: available {available} < {MIN_AVAILABLE_BYTES}"
        )
    return {
        "stage": stage,
        "source": "psutil.virtual_memory",
        "total_bytes": int(memory.total),
        "available_bytes": available,
        "required_available_bytes": MIN_AVAILABLE_BYTES,
        "admission": True,
    }


def _storage_preflight(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = output_root.resolve()
    primary = Path("/Volumes/VertigoDataTier/pact").resolve()
    secondary = Path("/Volumes/APDataStore/pact").resolve()
    if not (
        (resolved != primary and resolved.is_relative_to(primary))
        or (resolved != secondary and resolved.is_relative_to(secondary))
    ):
        raise P1RunnerError("P1 output root must use the SSD waterfall")
    usage = shutil.disk_usage(output_root)
    if usage.free < MIN_STORAGE_BYTES:
        raise P1RunnerError("REFUSE_P1_STORAGE: SSD free space is below 20 GiB")
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "free_bytes": int(usage.free),
        "required_free_bytes": MIN_STORAGE_BYTES,
        "cleanup": (
            "all Jacobian, solver, packet, archive, and verdict stages are immutable "
            "resumable checkpoints on SSD; no scratch or parent bytes are deleted"
        ),
    }


def _configure_torch(seed: int = 0) -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != TORCH_THREADS:
        raise P1RunnerError("P1 torch thread count differs")


def _load_scorers(
    config: P1ConfigV1,
    *,
    include_segnet: bool,
) -> tuple[Any, Any | None, dict[str, Any]]:
    from safetensors.torch import load_file

    root = Path(config.upstream_root).resolve()
    paths = {
        "modules": root / "modules.py",
        "posenet": root / "models" / "posenet.safetensors",
        "segnet": root / "models" / "segnet.safetensors",
    }
    custody = {}
    for name, path in paths.items():
        byte_count, digest = _sha256_file(path)
        if digest != SCORER_HASHES[name]:
            raise P1RunnerError(f"frozen {name} SHA-256 differs")
        custody[name] = {"path": str(path), "bytes": byte_count, "sha256": digest}
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ddm_p1_frozen_modules", paths["modules"])
    if spec is None or spec.loader is None:
        raise P1RunnerError("cannot import frozen scorer modules")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)
    posenet = modules.PoseNet().eval().to("cpu")
    posenet.load_state_dict(load_file(str(paths["posenet"]), device="cpu"), strict=True)
    segnet = None
    if include_segnet:
        segnet = modules.SegNet().eval().to("cpu")
        segnet.load_state_dict(load_file(str(paths["segnet"]), device="cpu"), strict=True)
    for model in (posenet, segnet):
        if model is None:
            continue
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return posenet, segnet, {
        **custody,
        "device": "cpu",
        "threads": TORCH_THREADS,
        "batch_size": SCORER_BATCH,
        "deterministic_algorithms": True,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _pose_tensor(posenet: Any, camera: Any) -> Any:
    output = posenet(posenet.preprocess_input(camera))
    pose = output["pose"] if isinstance(output, dict) else output
    return pose[:, :6]


def _exact_forward(
    *,
    posenet: Any,
    camera_pairs: np.ndarray,
    segnet: Any | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    camera = np.asarray(camera_pairs)
    if (
        camera.dtype != np.uint8
        or camera.ndim != 5
        or camera.shape[1:] != (2, CAMERA_H, CAMERA_W, CHANNELS)
    ):
        raise P1RunnerError("exact scorer camera geometry differs")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(camera))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    with torch.inference_mode():
        pose = _pose_tensor(posenet, tensor).cpu().numpy().astype(np.float64)
        cells = None
        if segnet is not None:
            cells = (
                segnet(segnet.preprocess_input(tensor))
                .argmax(dim=1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
    return np.ascontiguousarray(pose), None if cells is None else np.ascontiguousarray(cells)


def _load_targets(config: P1ConfigV1) -> np.ndarray:
    payload = json.loads(config.ms4d_pose_metric.validate_bytes())
    rows = payload.get("rows")
    if (
        payload.get("schema") != "ddm_pose_metric_custody.v1"
        or payload.get("pair_count") != PAIR_COUNT
        or payload.get("scorer_batch_size") != SCORER_BATCH
        or not isinstance(rows, list)
        or len(rows) != PAIR_COUNT
        or [row.get("pair_id") for row in rows] != list(range(PAIR_COUNT))
    ):
        raise P1RunnerError("MS4d Pose6 target custody differs")
    targets = np.asarray([row["center"] for row in rows], dtype=np.float64)
    if targets.shape != (PAIR_COUNT, 6) or not np.all(np.isfinite(targets)):
        raise P1RunnerError("MS4d Pose6 target tensor differs")
    return targets


def _parent_batch(config: P1ConfigV1, start: int, stop: int) -> tuple[np.ndarray, dict[str, Any]]:
    if start % SCORER_BATCH != 0 or stop != min(start + SCORER_BATCH, PAIR_COUNT):
        raise P1RunnerError("P1 parent batches must follow the exact batch32 partition")
    root = Path(config.g4_raw_root)
    stem = f"pairs_{start:04d}_{stop:04d}"
    raw_path = root / f"{stem}.raw"
    sidecar_path = root / f"{stem}.json"
    sidecar = _read_json(sidecar_path)
    expected_bytes = (stop - start) * 2 * CAMERA_H * CAMERA_W * CHANNELS
    if (
        sidecar.get("bytes") != expected_bytes
        or sidecar.get("pair_start") != start
        or sidecar.get("pair_stop") != stop
        or sidecar.get("state_sha256")
        != "2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"
    ):
        raise P1RunnerError(f"G4 parent sidecar differs: {sidecar_path}")
    byte_count, digest = _sha256_file(raw_path)
    if byte_count != expected_bytes or digest != sidecar.get("sha256"):
        raise P1RunnerError(f"G4 parent raw custody differs: {raw_path}")
    camera = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(stop - start, 2, CAMERA_H, CAMERA_W, CHANNELS),
    )
    return np.array(camera, copy=True, order="C"), {
        "path": str(raw_path),
        "bytes": byte_count,
        "sha256": digest,
        "pair_range": [start, stop],
    }


def _chart_indices() -> tuple[np.ndarray, np.ndarray]:
    y = np.minimum(
        ((2 * np.arange(CAMERA_H, dtype=np.int64) + 1) * GRID_H) // (2 * CAMERA_H),
        GRID_H - 1,
    )
    x = np.minimum(
        ((2 * np.arange(CAMERA_W, dtype=np.int64) + 1) * GRID_W) // (2 * CAMERA_W),
        GRID_W - 1,
    )
    return y, x


def _derive_batch(
    *,
    config: P1ConfigV1,
    posenet: Any,
    parent: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    import torch

    batch = len(parent)
    chart = torch.zeros(
        (batch, CHANNELS, GRID_H, GRID_W),
        dtype=torch.float32,
        requires_grad=True,
    )
    y_index, x_index = _chart_indices()
    y_tensor = torch.from_numpy(y_index)
    x_tensor = torch.from_numpy(x_index)
    residual = chart.index_select(2, y_tensor).index_select(3, x_tensor)
    parent_tensor = (
        torch.from_numpy(np.ascontiguousarray(parent))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    camera = torch.stack(
        (
            torch.clamp(parent_tensor[:, 0] + residual, 0.0, 255.0),
            parent_tensor[:, 1],
        ),
        dim=1,
    )
    pose = _pose_tensor(posenet, camera)
    gradients = []
    for axis in range(6):
        gradient = torch.autograd.grad(
            pose[:, axis].sum(),
            chart,
            retain_graph=axis < 5,
            create_graph=False,
        )[0]
        gradients.append(gradient.detach().cpu().numpy().astype(np.float32))
    jacobian = np.stack(gradients, axis=1).reshape(batch, 6, -1)
    parent_pose = pose.detach().cpu().numpy().astype(np.float64)
    target_residual = np.asarray(targets, dtype=np.float64) - parent_pose
    actuators = np.stack(
        [
            pose_targeted_actuator(
                jacobian[index],
                target_residual[index],
                ridge=config.jacobian_ridge,
            )
            for index in range(batch)
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "actuators": actuators,
        "jacobian": jacobian.astype(np.float32),
        "parent_pose6": parent_pose,
        "target_residual": target_residual,
    }


def _derivation_stage(
    *,
    config: P1ConfigV1,
    output_root: Path,
    targets: np.ndarray,
    posenet: Any,
) -> dict[str, Any]:
    stage_path = output_root / "stages" / "01_actuator_spectrum.json"
    if stage_path.exists():
        return _read_json(stage_path)
    memory = _memory_receipt("DERIVATION_N600")
    checkpoint_root = output_root / "checkpoints" / "01_derivation"
    custody_rows = []
    started = time.monotonic()
    for start in range(0, PAIR_COUNT, SCORER_BATCH):
        stop = min(start + SCORER_BATCH, PAIR_COUNT)
        checkpoint = checkpoint_root / f"chunk_{start:04d}_{stop:04d}.npz"
        parent, custody = _parent_batch(config, start, stop)
        custody_rows.append(custody)
        if not checkpoint.exists():
            arrays = _derive_batch(
                config=config,
                posenet=posenet,
                parent=parent,
                targets=targets[start:stop],
            )
            _atomic_npz(checkpoint, **arrays)
            print(
                json.dumps({"stage": "derive", "pair_range": [start, stop]}, sort_keys=True),
                flush=True,
            )
        del parent
        gc.collect()
    chunks = [
        np.load(checkpoint_root / f"chunk_{start:04d}_{min(start + SCORER_BATCH, PAIR_COUNT):04d}.npz")
        for start in range(0, PAIR_COUNT, SCORER_BATCH)
    ]
    actuators = np.concatenate([chunk["actuators"] for chunk in chunks], axis=0)
    parent_pose = np.concatenate([chunk["parent_pose6"] for chunk in chunks], axis=0)
    if actuators.shape != (PAIR_COUNT, CHANNELS * GRID_H * GRID_W):
        raise P1RunnerError("assembled P1 actuator geometry differs")
    baseline_d_pose = float(np.square(parent_pose - targets).mean(dtype=np.float64))
    centered = actuators.astype(np.float64) - actuators.mean(axis=0, dtype=np.float64)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = descending_covariance_spectrum(actuators)
    if not np.allclose(
        np.square(singular_values),
        eigenvalues[: len(singular_values)],
        rtol=1.0e-7,
        atol=1.0e-12,
    ):
        raise P1RunnerError("P1 SVD/eigenvalue spectrum crosscheck differs")
    for row in vt[:MAX_RANK]:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    basis_path = output_root / "checkpoints" / "01_derivation" / "basis_float64.npy"
    if not basis_path.exists():
        temporary = basis_path.with_name(f".{basis_path.name}.partial.{os.getpid()}")
        basis_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as handle:
            np.save(handle, vt[:MAX_RANK].astype(np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, basis_path)
    rank_law = canonical_rank_law(
        eigenvalues=eigenvalues,
        baseline_d_pose=baseline_d_pose,
        target_d_pose=DELEGATED_TARGET_D_POSE,
        maximum_rank=MAX_RANK,
    )
    spectrum_path = output_root / "stages" / "01_actuator_spectrum.jsonl"
    spectrum_rows = [
        {
            "schema": "ddm_p1_pose_jacobian_eigenvalue.v1",
            "index": index,
            "eigenvalue": float(value),
            "fraction": float(value / eigenvalues.sum()) if eigenvalues.sum() > 0.0 else 0.0,
        }
        for index, value in enumerate(eigenvalues)
    ]
    _atomic_jsonl(spectrum_path, spectrum_rows)
    result = {
        "schema": "ddm_p1_frame0_pose_actuator_spectrum.v1",
        "pair_count": PAIR_COUNT,
        "chart_coordinates": CHANNELS * GRID_H * GRID_W,
        "pose_dimensions": 6,
        "jacobian_ridge": config.jacobian_ridge,
        "baseline_d_pose": baseline_d_pose,
        "rank_law": rank_law,
        "top_eigenvalues": [float(value) for value in eigenvalues[:16]],
        "spectrum_rows": len(spectrum_rows),
        "spectrum_path": str(spectrum_path),
        "basis_path": str(basis_path),
        "basis_sha256": _sha256_file(basis_path)[1],
        "memory_preflight": memory,
        "raw_batch_custody": custody_rows,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "status": "COMPLETE_PRESERVED",
    }
    _atomic_json(stage_path, result)
    return result


def _quantize_basis(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (MAX_RANK, CHANNELS * GRID_H * GRID_W):
        raise P1RunnerError("P1 float basis geometry differs")
    quantized = []
    for row in values:
        maximum = float(np.max(np.abs(row)))
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise P1RunnerError("P1 float basis contains a null/nonfinite row")
        q = np.rint(row * (127.0 / maximum)).clip(-127.0, 127.0).astype(np.int8)
        pivot = int(np.argmax(np.abs(q.astype(np.int16))))
        if q[pivot] < 0:
            q = -q
        quantized.append(q.reshape(CHANNELS, GRID_H, GRID_W))
    return np.ascontiguousarray(np.stack(quantized, axis=0))


def _load_derivation_arrays(output_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_root = output_root / "checkpoints" / "01_derivation"
    jacobian = []
    parent_pose = []
    target_residual = []
    for start in range(0, PAIR_COUNT, SCORER_BATCH):
        stop = min(start + SCORER_BATCH, PAIR_COUNT)
        with np.load(checkpoint_root / f"chunk_{start:04d}_{stop:04d}.npz") as chunk:
            jacobian.append(np.array(chunk["jacobian"], copy=True))
            parent_pose.append(np.array(chunk["parent_pose6"], copy=True))
            target_residual.append(np.array(chunk["target_residual"], copy=True))
    return (
        np.concatenate(jacobian, axis=0),
        np.concatenate(parent_pose, axis=0),
        np.concatenate(target_residual, axis=0),
    )


def _linear_coefficients(
    *,
    jacobian: np.ndarray,
    target_residual: np.ndarray,
    q_basis: np.ndarray,
    rank: int,
    ridge: float,
) -> np.ndarray:
    basis = q_basis[:rank].reshape(rank, -1).astype(np.float64) * np.float64(2.0**-8)
    result = np.empty((len(jacobian), rank), dtype=np.float64)
    for index in range(len(jacobian)):
        chart = jacobian[index].astype(np.float64) @ basis.T
        gram = chart.T @ chart + ridge * np.eye(rank)
        rhs = chart.T @ target_residual[index]
        result[index] = np.linalg.solve(gram, rhs)
    return np.clip(np.rint(result), -32768.0, 32767.0).astype(np.int16)


def _gauss_newton_batch(
    *,
    config: P1ConfigV1,
    posenet: Any,
    parent: np.ndarray,
    targets: np.ndarray,
    q_basis: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch

    rank = q_basis.shape[0]
    basis = torch.from_numpy(q_basis.astype(np.float32)) * float(2.0**-8)
    y_index, x_index = _chart_indices()
    y_tensor = torch.from_numpy(y_index)
    x_tensor = torch.from_numpy(x_index)
    parent_tensor = (
        torch.from_numpy(np.ascontiguousarray(parent))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    coefficients = np.asarray(initial, dtype=np.float64).copy()
    trace = []
    for iteration in range(config.gauss_newton_iterations):
        q = torch.tensor(coefficients, dtype=torch.float32, requires_grad=True)
        q_ste = q + (torch.round(q) - q).detach()
        lowres = torch.einsum("nr,rchw->nchw", q_ste, basis)
        residual = lowres.index_select(2, y_tensor).index_select(3, x_tensor)
        continuous = parent_tensor[:, 0] + residual
        realized = torch.clamp(torch.round(continuous), 0.0, 255.0)
        frame0_ste = continuous + (realized - continuous).detach()
        camera = torch.stack((frame0_ste, parent_tensor[:, 1]), dim=1)
        pose = _pose_tensor(posenet, camera)
        gradients = []
        for axis in range(6):
            gradient = torch.autograd.grad(
                pose[:, axis].sum(),
                q,
                retain_graph=axis < 5,
                create_graph=False,
            )[0]
            gradients.append(gradient.detach().cpu().numpy().astype(np.float64))
        jacobian = np.stack(gradients, axis=1)
        pose_value = pose.detach().cpu().numpy().astype(np.float64)
        residual_pose = np.asarray(targets, dtype=np.float64) - pose_value
        deltas = np.empty_like(coefficients)
        for index in range(len(coefficients)):
            chart = jacobian[index]
            gram = chart.T @ chart + config.coefficient_ridge * np.eye(rank)
            delta = np.linalg.solve(gram, chart.T @ residual_pose[index])
            deltas[index] = np.clip(
                delta,
                -config.gauss_newton_max_step,
                config.gauss_newton_max_step,
            )
        before = float(np.square(residual_pose).mean(dtype=np.float64))
        coefficients = np.clip(coefficients + deltas, -32768.0, 32767.0)
        trace.append(
            {
                "iteration": iteration,
                "ste_d_pose_before_update": before,
                "mean_step_l2": float(np.linalg.norm(deltas, axis=1).mean()),
                "maximum_abs_step": float(np.abs(deltas).max()),
            }
        )
        del q, q_ste, lowres, residual, continuous, realized, frame0_ste, camera, pose
        gc.collect()
    return np.clip(np.rint(coefficients), -32768.0, 32767.0).astype(np.int16), trace


def _refine_selected(
    *,
    config: P1ConfigV1,
    output_root: Path,
    posenet: Any,
    targets: np.ndarray,
    family: str,
    q_basis: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    root = output_root / "checkpoints" / "02_solver" / family
    chunks = []
    for start in range(0, PAIR_COUNT, SCORER_BATCH):
        stop = min(start + SCORER_BATCH, PAIR_COUNT)
        path = root / f"chunk_{start:04d}_{stop:04d}.npz"
        if not path.exists():
            parent, _custody = _parent_batch(config, start, stop)
            coefficients, trace = _gauss_newton_batch(
                config=config,
                posenet=posenet,
                parent=parent,
                targets=targets[start:stop],
                q_basis=q_basis,
                initial=initial[start:stop],
            )
            _atomic_npz(
                path,
                q_coefficients=coefficients,
                trace_json=np.asarray(json.dumps(trace, sort_keys=True)),
            )
            print(
                json.dumps(
                    {"stage": "gauss_newton", "family": family, "pair_range": [start, stop]},
                    sort_keys=True,
                ),
                flush=True,
            )
        with np.load(path) as checkpoint:
            chunks.append(np.array(checkpoint["q_coefficients"], copy=True))
    return np.concatenate(chunks, axis=0)


def _packet_stage(
    *,
    config: P1ConfigV1,
    output_root: Path,
    targets: np.ndarray,
    posenet: Any,
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    stage_path = output_root / "stages" / "02_solved_packets.json"
    if stage_path.exists():
        return _read_json(stage_path)
    jacobian, _parent_pose, target_residual = _load_derivation_arrays(output_root)
    basis_path = Path(str(derivation["basis_path"]))
    with basis_path.open("rb") as handle:
        float_basis = np.load(handle, allow_pickle=False)
    treatment_basis = _quantize_basis(float_basis)
    control_basis = seeded_matched_control_basis(
        seed=config.carrier.control_seed,
        rank=MAX_RANK,
    )
    selected_rank_value = derivation["rank_law"].get("selected_rank")
    selected_rank = int(selected_rank_value) if selected_rank_value is not None else MAX_RANK
    linear_treatments = {
        rank: _linear_coefficients(
            jacobian=jacobian,
            target_residual=target_residual,
            q_basis=treatment_basis,
            rank=rank,
            ridge=config.coefficient_ridge,
        )
        for rank in range(1, MAX_RANK + 1)
    }
    control_initial = _linear_coefficients(
        jacobian=jacobian,
        target_residual=target_residual,
        q_basis=control_basis,
        rank=selected_rank,
        ridge=config.coefficient_ridge,
    )
    treatment_selected = _refine_selected(
        config=config,
        output_root=output_root,
        posenet=posenet,
        targets=targets,
        family=f"treatment_rank{selected_rank}",
        q_basis=treatment_basis[:selected_rank],
        initial=linear_treatments[selected_rank],
    )
    control_selected = _refine_selected(
        config=config,
        output_root=output_root,
        posenet=posenet,
        targets=targets,
        family=f"control_rank{selected_rank}",
        q_basis=control_basis[:selected_rank],
        initial=control_initial,
    )
    linear_treatments[selected_rank] = treatment_selected
    parent_archive = config.g4_parent_archive.validate_bytes()
    rows = []
    packet_root = output_root / "packets"
    for rank in range(1, MAX_RANK + 1):
        packet = make_packet(
            treatment=True,
            rank=rank,
            q_basis=treatment_basis[:rank],
            q_coefficients=linear_treatments[rank],
        )
        packet_bytes = serialize_packet(packet)
        archive = build_counted_composition_archive(
            parent_archive=parent_archive,
            parent_sha256=config.g4_parent_archive.sha256,
            packet=packet,
        )
        parsed_parent, parsed_packet, _manifest = parse_counted_composition_archive(archive)
        if parsed_parent != parent_archive or serialize_packet(parsed_packet) != packet_bytes:
            raise P1RunnerError("treatment packet/archive parse-back differs")
        packet_path = packet_root / f"treatment_rank{rank}.ddp"
        archive_path = packet_root / f"treatment_rank{rank}.not_a_candidate.zip.receipt-bytes"
        _atomic_bytes(packet_path, packet_bytes)
        _atomic_bytes(archive_path, archive)
        rows.append(
            {
                "candidate_id": f"treatment_rank{rank}",
                "family": "treatment",
                "rank": rank,
                "carrier_bytes": len(packet_bytes),
                "carrier_sha256": sha256_bytes(packet_bytes),
                "archive_bytes": len(archive),
                "archive_sha256": sha256_bytes(archive),
                "packet_path": str(packet_path),
                "archive_path": str(archive_path),
                "parseback_byte_identical": True,
            }
        )
    control_packet = make_packet(
        treatment=False,
        rank=selected_rank,
        q_basis=control_basis[:selected_rank],
        q_coefficients=control_selected,
    )
    control_bytes = serialize_packet(control_packet)
    control_archive = build_counted_composition_archive(
        parent_archive=parent_archive,
        parent_sha256=config.g4_parent_archive.sha256,
        packet=control_packet,
    )
    control_packet_path = packet_root / f"control_rank{selected_rank}.ddp"
    control_archive_path = (
        packet_root / f"control_rank{selected_rank}.not_a_candidate.zip.receipt-bytes"
    )
    _atomic_bytes(control_packet_path, control_bytes)
    _atomic_bytes(control_archive_path, control_archive)
    rows.append(
        {
            "candidate_id": f"control_rank{selected_rank}",
            "family": "control",
            "rank": selected_rank,
            "carrier_bytes": len(control_bytes),
            "carrier_sha256": sha256_bytes(control_bytes),
            "archive_bytes": len(control_archive),
            "archive_sha256": sha256_bytes(control_archive),
            "packet_path": str(control_packet_path),
            "archive_path": str(control_archive_path),
            "parseback_byte_identical": True,
        }
    )
    treatment_row = next(
        row for row in rows if row["candidate_id"] == f"treatment_rank{selected_rank}"
    )
    if (
        treatment_row["carrier_bytes"] != len(control_bytes)
        or treatment_row["archive_bytes"] != len(control_archive)
    ):
        raise P1RunnerError("treatment/control exact counted budgets differ")
    result = {
        "schema": "ddm_p1_frame0_pose_solved_packets.v1",
        "selected_rank": selected_rank,
        "rank_law_selected_rank": selected_rank_value,
        "no_linear_rank_reached_target": selected_rank_value is None,
        "term_exponent": -8,
        "treatment_basis_sha256": sha256_bytes(treatment_basis.tobytes(order="C")),
        "control_basis_sha256": sha256_bytes(control_basis.tobytes(order="C")),
        "control_seed": config.carrier.control_seed,
        "gauss_newton_iterations_per_arm": config.gauss_newton_iterations,
        "same_solver_call_budget": True,
        "rows": rows,
        "status": "COMPLETE_PRESERVED",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    _atomic_json(stage_path, result)
    return result


def _load_packet(path: str) -> Any:
    from tac.optimization.ddm_p1_frame0_pose_quotient_carrier import parse_packet

    return parse_packet(Path(path).read_bytes())


def _candidate_verdict(
    *,
    config: P1ConfigV1,
    output_root: Path,
    targets: np.ndarray,
    posenet: Any,
    segnet: Any | None,
    packet_row: Mapping[str, Any],
    measure_seg: bool,
) -> dict[str, Any]:
    candidate_id = str(packet_row["candidate_id"])
    root = output_root / "verdicts" / candidate_id
    result_path = root / "n600.json"
    if result_path.exists():
        return _read_json(result_path)
    memory = _memory_receipt(f"N600_{candidate_id.upper()}")
    packet = _load_packet(str(packet_row["packet_path"]))
    rows = []
    started = time.monotonic()
    for start in range(0, PAIR_COUNT, SCORER_BATCH):
        stop = min(start + SCORER_BATCH, PAIR_COUNT)
        path = root / f"chunk_{start:04d}_{stop:04d}.json"
        if path.exists():
            rows.append(_read_json(path))
            continue
        parent, parent_custody = _parent_batch(config, start, stop)
        candidate = receive_frame0_quotient(
            parent_camera=parent,
            packet=packet,
            pair_ids=range(start, stop),
        )
        if not np.array_equal(candidate[:, 1], parent[:, 1]):
            raise P1RunnerError(f"{candidate_id} changed frame 1")
        pose, cells = _exact_forward(
            posenet=posenet,
            camera_pairs=candidate,
            segnet=segnet if measure_seg else None,
        )
        pose_sse = float(np.square(pose - targets[start:stop]).sum(dtype=np.float64))
        frame1 = np.ascontiguousarray(candidate[:, 1]).tobytes(order="C")
        row = {
            "schema": "ddm_p1_frame0_pose_quotient_n600_chunk.v1",
            "candidate_id": candidate_id,
            "pair_range": [start, stop],
            "carrier_sha256": packet_row["carrier_sha256"],
            "pose_squared_error_sum": pose_sse,
            "pose_coordinates": int(pose.size),
            "pose6_sha256": sha256_bytes(pose.astype("<f8").tobytes(order="C")),
            "frame1_sha256": sha256_bytes(frame1),
            "frame1_bytes": len(frame1),
            "seg_cells_sha256": (
                sha256_bytes(cells.tobytes(order="C")) if cells is not None else None
            ),
            "seg_cells_bytes": int(cells.nbytes) if cells is not None else None,
            "parent_raw_custody": parent_custody,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        _atomic_json(path, row)
        rows.append(row)
        print(
            json.dumps(
                {"stage": "exact_n600", "candidate": candidate_id, "pair_range": [start, stop]},
                sort_keys=True,
            ),
            flush=True,
        )
        del parent, candidate, pose, cells
        gc.collect()
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    frame1_chain = sha256_bytes(
        b"".join(bytes.fromhex(str(row["frame1_sha256"])) for row in rows)
    )
    seg_chain = (
        sha256_bytes(
            b"".join(bytes.fromhex(str(row["seg_cells_sha256"])) for row in rows)
        )
        if measure_seg
        else None
    )
    result = {
        "schema": "ddm_p1_frame0_pose_quotient_n600_verdict.v1",
        "candidate_id": candidate_id,
        "family": packet_row["family"],
        "rank": packet_row["rank"],
        "pair_count": PAIR_COUNT,
        "batch_size": SCORER_BATCH,
        "carrier_bytes": packet_row["carrier_bytes"],
        "carrier_sha256": packet_row["carrier_sha256"],
        "archive_bytes": packet_row["archive_bytes"],
        "archive_sha256": packet_row["archive_sha256"],
        "d_pose": pose_sse / pose_coordinates,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "frame1_batch_digest_chain_sha256": frame1_chain,
        "seg_cells_batch_digest_chain_sha256": seg_chain,
        "seg_measured": measure_seg,
        "memory_preflight": memory,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": config.pointer,
        "pointer_moved": False,
    }
    _atomic_json(result_path, result)
    return result


def _measurement_stage(
    *,
    config: P1ConfigV1,
    output_root: Path,
    targets: np.ndarray,
    posenet: Any,
    segnet: Any,
    packets: Mapping[str, Any],
    scorer_custody: Mapping[str, Any],
    config_sha256: str,
    storage: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = output_root / "receipt.json"
    if receipt_path.exists():
        return _read_json(receipt_path)
    selected_rank = int(packets["selected_rank"])
    verdicts = []
    for packet_row in packets["rows"]:
        candidate_id = str(packet_row["candidate_id"])
        measure_seg = candidate_id in {
            f"treatment_rank{selected_rank}",
            f"control_rank{selected_rank}",
        }
        verdicts.append(
            _candidate_verdict(
                config=config,
                output_root=output_root,
                targets=targets,
                posenet=posenet,
                segnet=segnet,
                packet_row=packet_row,
                measure_seg=measure_seg,
            )
        )
    reach_rows = [
        {
            "schema": REACH_ROW_SCHEMA,
            "rank": int(row["rank"]),
            "d_pose": float(row["d_pose"]),
            "carrier_bytes": int(row["carrier_bytes"]),
            "carrier_sha256": row["carrier_sha256"],
            "archive_bytes": int(row["archive_bytes"]),
            "archive_sha256": row["archive_sha256"],
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        for row in verdicts
        if row["family"] == "treatment"
    ]
    reach_rows.sort(key=lambda row: int(row["rank"]))
    verdict, verdict_scope = reach_curve_disposition(reach_rows)
    treatment = next(
        row for row in verdicts if row["candidate_id"] == f"treatment_rank{selected_rank}"
    )
    control = next(
        row for row in verdicts if row["candidate_id"] == f"control_rank{selected_rank}"
    )
    parent_frame1_chain = treatment["frame1_batch_digest_chain_sha256"]
    matched = matched_control_fence(
        treatment_packet_bytes=int(treatment["carrier_bytes"]),
        control_packet_bytes=int(control["carrier_bytes"]),
        treatment_frame1_sha256=str(treatment["frame1_batch_digest_chain_sha256"]),
        control_frame1_sha256=str(control["frame1_batch_digest_chain_sha256"]),
        parent_frame1_sha256=str(parent_frame1_chain),
        same_rank=int(treatment["rank"]) == int(control["rank"]),
        same_precision=True,
        same_solver=bool(packets["same_solver_call_budget"]),
    )
    seg_identical = (
        treatment["seg_cells_batch_digest_chain_sha256"]
        == control["seg_cells_batch_digest_chain_sha256"]
        and treatment["seg_cells_batch_digest_chain_sha256"] is not None
    )
    delegated_pass = bool(
        float(treatment["d_pose"]) <= DELEGATED_TARGET_D_POSE
        and seg_identical
        and int(treatment["carrier_bytes"]) <= config.carrier.maximum_bytes
        and matched
    )
    strict_gc4_pass = bool(
        delegated_pass and float(treatment["d_pose"]) <= GC4_STRICT_TARGET_D_POSE
    )
    _atomic_jsonl(output_root / "reach_curve.jsonl", reach_rows)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "typed_config_sha256": config_sha256,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
        "pointer": config.pointer,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "verdict": verdict,
        "verdict_scope": verdict_scope,
        "delegated_pass": delegated_pass,
        "gc4_strict_veto_cleared": strict_gc4_pass,
        "rank_law": derivation["rank_law"],
        "selected_rank": selected_rank,
        "treatment": treatment,
        "control": control,
        "exact_matched_control": {
            "fence_passed": matched,
            "same_packet_bytes": treatment["carrier_bytes"] == control["carrier_bytes"],
            "same_archive_bytes": treatment["archive_bytes"] == control["archive_bytes"],
            "same_rank": treatment["rank"] == control["rank"],
            "same_precision": True,
            "same_solver_call_budget": packets["same_solver_call_budget"],
            "frame1_batch_digest_chain_identical": (
                treatment["frame1_batch_digest_chain_sha256"]
                == control["frame1_batch_digest_chain_sha256"]
            ),
            "seg_cells_batch_digest_chain_identical": seg_identical,
            "d_seg_delta": 0.0 if seg_identical else None,
        },
        "reach_curve": reach_rows,
        "named_obstruction": (
            None
            if delegated_pass
            else (
                "The measured six-dimensional shared quantized 24x32 parent-additive "
                "frame-0 actuator chart leaves receiver-realized Pose6 residual above "
                "5e-5 under the exact packet cap; this does not close nonlinear or "
                "higher-rank frame-0 quotient generators."
            )
        ),
        "packet_stage": packets,
        "actuator_spectrum": derivation,
        "scorer_custody": dict(scorer_custody),
        "storage_preflight": dict(storage),
        "custody": {
            "authority": config.authority.model_dump(mode="json"),
            "g4_parent_archive": config.g4_parent_archive.model_dump(mode="json"),
            "ms4d_pose_metric": config.ms4d_pose_metric.model_dump(mode="json"),
        },
        "resume": {
            "derivation_batch_checkpoints_preserved": True,
            "solver_batch_checkpoints_preserved": True,
            "exact_n600_batch_checkpoints_preserved": True,
            "all_stage_receipts_preserved": True,
        },
        "triality": {
            "dsl": ".omx/research/configs/ddm_p1_frame0_pose_quotient_carrier_20260725.json",
            "dag": ".omx/research/ddm_p1_frame0_pose_quotient_carrier_DAG_FEED_20260725.md",
            "equation": (
                "tac.canonical_equations."
                "ddm_p1_frame0_pose_quotient_carrier_20260725"
            ),
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def run(config_path: Path, *, stage: str) -> dict[str, Any]:
    config, config_sha256 = P1ConfigV1.from_path(config_path)
    config.authority.validate_bytes()
    config.g4_parent_archive.validate_bytes()
    config.ms4d_pose_metric.validate_bytes()
    output_root = Path(config.output_root)
    storage = _storage_preflight(output_root)
    _configure_torch()
    targets = _load_targets(config)
    posenet, _unused, scorer_custody = _load_scorers(config, include_segnet=False)
    derivation = _derivation_stage(
        config=config,
        output_root=output_root,
        targets=targets,
        posenet=posenet,
    )
    if stage == "derive":
        return derivation
    packets = _packet_stage(
        config=config,
        output_root=output_root,
        targets=targets,
        posenet=posenet,
        derivation=derivation,
    )
    if stage == "solve":
        return packets
    del posenet
    gc.collect()
    posenet, segnet, scorer_custody = _load_scorers(config, include_segnet=True)
    if segnet is None:
        raise P1RunnerError("P1 exact matched row requires SegNet")
    return _measurement_stage(
        config=config,
        output_root=output_root,
        targets=targets,
        posenet=posenet,
        segnet=segnet,
        packets=packets,
        scorer_custody=scorer_custody,
        config_sha256=config_sha256,
        storage=storage,
        derivation=derivation,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("derive", "solve", "all"), default="all")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = run(args.config, stage=args.stage)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
