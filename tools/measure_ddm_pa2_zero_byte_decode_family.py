#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure PA2 zero-byte decode transforms on the three sealed control bases.

This runner is intentionally checkpoint-first.  Every frozen-scorer batch is
an immutable stage, every greedy decision is reconstructed from those stages,
and selected receiver output is preserved on the SSD tier one batch at a time.
No counted archive member is created or changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_ms2r_tolerance_capped_solve_r2 import (  # noqa: E402
    quantize_uint8_half_up,
)
from tac.optimization.ddm_pa2_zero_byte_decode_family import (  # noqa: E402
    BLOCKERS,
    PA2Member,
    apply_stack,
    family_inventory,
    scorer_resize,
)
from tac.optimization.ddm_runtime_receiver import (  # noqa: E402
    PA1_TRANSFORM_ID,
    _apply_pa1_frame0_affine,
    _derive_pa1_affine,
    _merge_pose_moments,
    _pose_moment_row,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

SCHEMA = "ddm_pa2_zero_byte_decode_family_measurement.v1"
CONFIG_SCHEMA = "DDMPA2ZeroByteDecodeFamilyConfigV1"
ROW_SCHEMA = "ddm_pa2_zero_byte_decode_family_batch32_row.v1"
RUN_ID = "ddm_pa2_zero_byte_decode_family_20260724T194836Z"
LANE_ID = "lane_ddm_pa2_zero_byte_decode_family_20260724"
AXIS = "[macOS-CPU frozen-scorer advisory]"
CAMERA_SHAPE = (600, 2, 874, 1164, 3)
SEG_SHAPE = (600, 384, 512)
PA1_MEMBER = PA1_TRANSFORM_ID
STRICT_CANDIDATES = (
    PA2Member.SPATIAL_STEM_RESIDUAL.value,
    PA2Member.TEMPORAL_XIHAT_FRAME0.value,
    PA2Member.TEMPORAL_XIHAT_FRAME1.value,
)


class PA2MeasurementError(RuntimeError):
    """A bound artifact, checkpoint, or deterministic replay differed."""


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    base_id: Literal["IC1_W_joint_PA1", "IC2_W_seg_PA1", "MS2R_q4_q8"]
    kind: Literal["raw", "ms2r"]
    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    custody_receipt_path: str
    custody_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoded_raw_path: str | None = None
    decoded_raw_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    c1_root: str | None = None


class DDMPA2ZeroByteDecodeFamilyConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMPA2ZeroByteDecodeFamilyConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_pa2_zero_byte_decode_family_20260724T194836Z"] = RUN_ID
    lane_id: Literal["lane_ddm_pa2_zero_byte_decode_family_20260724"] = LANE_ID
    scorer_config_path: str
    scorer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ms2r_receipt_path: str
    ms2r_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bases: tuple[BaseConfig, BaseConfig, BaseConfig]
    bulk_root: str
    receipt_root: str
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[32] = 32
    scorer_threads: Literal[4] = 4
    seed: int = 1234
    minimum_free_bytes: int = Field(ge=12_000_000_000)
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    main_review_required: Literal[True] = True
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"


class FrozenScorerConfig(BaseModel):
    """The strict scorer fields consumed from the sealed V14 config."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    schema_: Literal["DDMV14RealizationFidelityConfigV1"] = Field(alias="schema")
    seed: Literal[1234]
    pair_count: Literal[600]
    target_cache_path: str
    target_cache_bytes: int
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_root: str
    scorer_threads: Literal[4]
    scorer_batch_size: Literal[16]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _bound(path_value: str, sha256: str, label: str) -> Path:
    path = _resolve(path_value)
    if not path.is_file():
        raise PA2MeasurementError(f"{label} is absent: {path}")
    if _sha256_file(path)[1] != sha256:
        raise PA2MeasurementError(f"{label} SHA-256 differs")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise PA2MeasurementError(f"malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise PA2MeasurementError(f"JSON root must be an object: {path}")
    return value


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = rfc8785_canonicalize(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PA2MeasurementError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = _sha256_file(path)
        expected = (len(payload), hashlib.sha256(payload).hexdigest())
        if observed != expected:
            raise PA2MeasurementError(f"immutable stage differs: {path}")
        return
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_config(path: Path) -> tuple[DDMPA2ZeroByteDecodeFamilyConfigV1, str]:
    payload = path.read_bytes()
    value = DDMPA2ZeroByteDecodeFamilyConfigV1.model_validate_json(payload, strict=True)
    ids = [base.base_id for base in value.bases]
    if ids != ["IC1_W_joint_PA1", "IC2_W_seg_PA1", "MS2R_q4_q8"]:
        raise PA2MeasurementError("three bases must appear in canonical IC1/IC2/MS2R order")
    for base in value.bases:
        if base.kind == "raw" and (
            base.decoded_raw_path is None
            or base.decoded_raw_sha256 is None
            or base.c1_root is not None
        ):
            raise PA2MeasurementError(f"{base.base_id} raw custody fields differ")
        if base.kind == "ms2r" and (
            base.c1_root is None
            or base.decoded_raw_path is not None
            or base.decoded_raw_sha256 is not None
        ):
            raise PA2MeasurementError("MS2R source fields differ")
    return value, hashlib.sha256(payload).hexdigest()


def _load_models(config: FrozenScorerConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Load the exact frozen CPU scorers without importing carrier tooling."""

    import torch
    from safetensors.torch import load_file

    upstream = Path(config.upstream_root).resolve()
    modules_path = upstream / "modules.py"
    if not modules_path.is_file():
        raise PA2MeasurementError("frozen scorer modules.py is absent")
    spec = importlib.util.spec_from_file_location("ddm_pa2_upstream_modules", modules_path)
    if spec is None or spec.loader is None:
        raise PA2MeasurementError("cannot import frozen scorer modules")
    modules = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(upstream))
    try:
        spec.loader.exec_module(modules)
    finally:
        if sys.path[0] == str(upstream):
            sys.path.pop(0)
    torch.set_num_threads(config.scorer_threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    segnet = modules.SegNet().eval().cpu()
    posenet = modules.PoseNet().eval().cpu()
    seg_path = Path(modules.segnet_sd_path).resolve()
    pose_path = Path(modules.posenet_sd_path).resolve()
    segnet.load_state_dict(load_file(str(seg_path), device="cpu"), strict=True)
    posenet.load_state_dict(load_file(str(pose_path), device="cpu"), strict=True)
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return segnet, posenet, {
        "modules_path": str(modules_path),
        "modules_sha256": _sha256_file(modules_path)[1],
        "segnet_weights_path": str(seg_path),
        "segnet_weights_sha256": _sha256_file(seg_path)[1],
        "posenet_weights_path": str(pose_path),
        "posenet_weights_sha256": _sha256_file(pose_path)[1],
        "device": "cpu",
        "threads": config.scorer_threads,
        "batch_size": 32,
        "deterministic_algorithms": True,
        "evidence_axis": AXIS,
    }


def _forward(segnet: Any, posenet: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    value = np.asarray(camera_pairs)
    if value.dtype != np.uint8 or value.ndim != 5 or value.shape[1:] != CAMERA_SHAPE[1:]:
        raise PA2MeasurementError("scorer requires uint8 [B,2,874,1164,3]")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    with torch.inference_mode():
        cells = (
            segnet(segnet.preprocess_input(tensor))
            .argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        pose_output = posenet(posenet.preprocess_input(tensor))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def _source_chunk(c1_root: Path, chunk_index: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    base = c1_root / "prepare_chunks" / f"chunk-{chunk_index:04d}"
    manifest = _read_json(base.with_suffix(".manifest.json"))
    pair_ids = manifest.get("pair_ids")
    count = manifest.get("pair_count")
    if (
        manifest.get("complete") is not True
        or not isinstance(pair_ids, list)
        or type(count) is not int
        or len(pair_ids) != count
    ):
        raise PA2MeasurementError("C1 source chunk manifest differs")
    shape = (count, 384, 512, 3)
    y0 = np.fromfile(base.with_suffix(".y0.bin"), dtype=np.uint8).reshape(shape)
    y1 = np.fromfile(base.with_suffix(".y1.bin"), dtype=np.uint8).reshape(shape)
    return [int(value) for value in pair_ids], y0, y1


def _source_batch(c1_root: Path, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rows0: list[np.ndarray] = []
    rows1: list[np.ndarray] = []
    chunks: dict[int, tuple[list[int], np.ndarray, np.ndarray]] = {}
    for pair_id in range(start, start + count):
        chunk_index, local = divmod(pair_id, 12)
        if chunk_index not in chunks:
            chunks[chunk_index] = _source_chunk(c1_root, chunk_index)
        pair_ids, y0, y1 = chunks[chunk_index]
        if pair_ids[local] != pair_id:
            raise PA2MeasurementError("C1 source pair sequence differs")
        rows0.append(y0[local])
        rows1.append(y1[local])
    return np.stack(rows0), np.stack(rows1)


def _advisory_objective(*, errors: int, sites: int, d_pose: float, bytes_: int) -> float:
    return (
        100.0 * (errors / sites)
        + math.sqrt(10.0 * d_pose)
        + 25.0 * bytes_ / 37_545_489
    )


class BaseLoader:
    """Deterministic batch32 camera loader for one exact sealed base."""

    def __init__(
        self,
        base: BaseConfig,
        *,
        selected_steps: Sequence[int],
    ) -> None:
        self.base = base
        self._selected_steps = tuple(int(value) for value in selected_steps)
        self._raw: np.memmap | None = None
        self._operator: DisjointResizeOperator | None = None
        if base.kind == "raw":
            assert base.decoded_raw_path is not None
            self._raw = np.memmap(
                Path(base.decoded_raw_path),
                mode="r",
                dtype=np.uint8,
                shape=CAMERA_SHAPE,
            )
        else:
            if len(self._selected_steps) != 600 or any(
                value not in (4, 8) for value in self._selected_steps
            ):
                raise PA2MeasurementError("MS2R selected-step inventory differs")
            self._operator = DisjointResizeOperator.build(
                camera_h=874,
                camera_w=1164,
                scorer_h=384,
                scorer_w=512,
            )

    def batch(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= 600 or stop - start > 32:
            raise PA2MeasurementError("base batch range differs")
        if self._raw is not None:
            return np.array(self._raw[start:stop], copy=True, order="C")
        assert self.base.c1_root is not None and self._operator is not None
        y0, y1 = _source_batch(Path(self.base.c1_root), start, stop - start)
        camera = np.empty((stop - start, 2, 874, 1164, 3), dtype=np.uint8)
        for local, pair_id in enumerate(range(start, stop)):
            step = self._selected_steps[pair_id]
            q0 = quantize_uint8_half_up(y0[local], step)
            q1 = quantize_uint8_half_up(y1[local], step)
            camera[local, 0] = realize_factor2_uint8_scorer_plane(self._operator, q0)
            camera[local, 1] = realize_factor2_uint8_scorer_plane(self._operator, q1)
        for frame_id, target in ((0, q0), (1, q1)):
            verification = verify_factor2_uint8_scorer_plane(
                self._operator,
                camera[-1, frame_id],
                target,
            )
            if not verification.certified_exact:
                raise PA2MeasurementError("MS2R factor-2 receiver parse-back differs")
        return camera


def _apply_receiver_stack(
    camera: np.ndarray,
    stack: Sequence[str],
    *,
    pa1_affine: tuple[Any, Any] | None,
) -> np.ndarray:
    output = camera
    pending: list[str] = []
    for member in stack:
        if member == PA1_MEMBER:
            if pending:
                output = apply_stack(output, pending)
                pending.clear()
            if pa1_affine is None:
                raise PA2MeasurementError("PA1 affine is absent")
            import torch

            output = (
                _apply_pa1_frame0_affine(
                    torch.from_numpy(np.ascontiguousarray(output)),
                    pa1_affine[0],
                    pa1_affine[1],
                )
                .cpu()
                .numpy()
            )
        else:
            pending.append(member)
    if pending:
        output = apply_stack(output, pending)
    return np.ascontiguousarray(output)


def _pa1_affine(
    loader: BaseLoader,
    *,
    checkpoint_root: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    path = checkpoint_root / "pa1_decoded_moments_and_affine.json"
    rows: list[dict[str, Any]] = []
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        camera = loader.batch(start, stop)
        import torch

        rows.append(_pose_moment_row(torch.from_numpy(camera)))
        del camera
        gc.collect()
    moments = _merge_pose_moments(rows)
    gain, bias = _derive_pa1_affine(moments)
    value = {
        "schema": "ddm_pa2_pa1_decoded_affine.v1",
        "transform_id": PA1_MEMBER,
        "rate_class": "FREE",
        "payload_bytes": 0,
        "moments": moments,
        "gain_f32": gain.tolist(),
        "bias_f32": bias.tolist(),
        "derivation": "decoded PoseNet YUV6 moments x frozen first-stem BN target",
        "score_claim": False,
    }
    _publish_json(path, value)
    return gain, bias, value


def _metric_row(
    *,
    cells: np.ndarray,
    poses: np.ndarray,
    labels: np.ndarray,
    target_poses: np.ndarray,
) -> dict[str, Any]:
    return {
        "errors": int(np.count_nonzero(cells != labels)),
        "sites": int(cells.size),
        "pose_squared_error_sum": float(
            np.square(poses - target_poses).sum(dtype=np.float64)
        ),
        "pose_coordinates": int(poses.size),
    }


def _score_stack(
    *,
    loader: BaseLoader,
    base: BaseConfig,
    stack: Sequence[str],
    pa1_affine: tuple[Any, Any] | None,
    labels: np.memmap,
    target_poses: np.memmap,
    segnet: Any,
    posenet: Any,
    checkpoint_root: Path,
) -> dict[str, Any]:
    stack_id = (
        "BASE"
        if not stack
        else hashlib.sha256(
            rfc8785_canonicalize({"stack": list(stack)})
        ).hexdigest()[:16]
    )
    rows: list[dict[str, Any]] = []
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        path = checkpoint_root / "scorer" / stack_id / f"batch_{start:04d}_{stop:04d}.json"
        source = loader.batch(start, stop)
        source_sha256 = hashlib.sha256(source.tobytes()).hexdigest()
        camera = _apply_receiver_stack(
            source,
            stack,
            pa1_affine=pa1_affine,
        )
        camera_sha256 = hashlib.sha256(camera.tobytes()).hexdigest()
        if path.exists():
            row = _read_json(path)
            if (
                row.get("schema") != ROW_SCHEMA
                or row.get("base_id") != base.base_id
                or row.get("stack") != list(stack)
                or row.get("pair_range") != [start, stop]
                or row.get("source_camera_sha256") != source_sha256
                or row.get("receiver_camera_sha256") != camera_sha256
            ):
                raise PA2MeasurementError(f"preserved scorer row differs: {path}")
        else:
            cells, poses = _forward(segnet, posenet, camera)
            row = {
                "schema": ROW_SCHEMA,
                "base_id": base.base_id,
                "stack": list(stack),
                "pair_range": [start, stop],
                "source_camera_sha256": source_sha256,
                "receiver_camera_sha256": camera_sha256,
                **_metric_row(
                    cells=cells,
                    poses=poses,
                    labels=np.asarray(labels[start:stop], dtype=np.uint8),
                    target_poses=np.asarray(target_poses[start:stop], dtype=np.float64),
                ),
                "scorer_batch_size": 32,
                "evidence_axis": AXIS,
                "score_claim": False,
            }
            _publish_json(path, row)
        rows.append(row)
        print(
            f"[PA2] {base.base_id} {stack_id} {start:04d}:{stop:04d}",
            flush=True,
        )
        del source, camera
        gc.collect()
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = math.fsum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    objective = _advisory_objective(
        errors=errors,
        sites=sites,
        d_pose=d_pose,
        bytes_=base.archive_bytes,
    )
    return {
        "stack": list(stack),
        "errors": errors,
        "sites": sites,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "advisory_objective_at_exact_archive_bytes": objective,
        "archive_bytes": base.archive_bytes,
        "scorer_batch_size": 32,
        "stage_count": len(rows),
    }


def _blind_identity(
    *,
    loader: BaseLoader,
    base: BaseConfig,
    checkpoint_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        path = checkpoint_root / "blind_identity" / f"batch_{start:04d}_{stop:04d}.json"
        source = loader.batch(start, stop)
        output = apply_stack(source, (PA2Member.BLIND_ZERO_FILL.value,))
        before = scorer_resize(source)
        after = scorer_resize(output)
        exact = bool(np.array_equal(before.cpu().numpy(), after.cpu().numpy()))
        row = {
            "schema": "ddm_pa2_blind_input_identity_batch32.v1",
            "base_id": base.base_id,
            "pair_range": [start, stop],
            "source_camera_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "receiver_camera_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "seg_pose_shared_resize_bit_identical": exact,
            "archive_bytes_saved_on_pure_generator_base": 0,
            "score_claim": False,
        }
        if not exact:
            raise PA2MeasurementError("#401 changed the frozen scorer input")
        _publish_json(path, row)
        rows.append(row)
        del source, output, before, after
        gc.collect()
    return {
        "member": PA2Member.BLIND_ZERO_FILL.value,
        "n600_batch32_exact": all(
            row["seg_pose_shared_resize_bit_identical"] for row in rows
        ),
        "stage_count": len(rows),
        "conditional_delta_d_seg": 0.0,
        "conditional_delta_d_pose": 0.0,
        "conditional_delta_archive_bytes": 0,
        "verdict": "EXECUTED_SCORE_NEUTRAL_ON_PURE_GENERATOR_BASE",
    }


def _materialize_selected(
    *,
    loader: BaseLoader,
    base: BaseConfig,
    stack: Sequence[str],
    pa1_affine: tuple[Any, Any] | None,
    checkpoint_root: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    total = 0
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        camera = _apply_receiver_stack(
            loader.batch(start, stop),
            stack,
            pa1_affine=pa1_affine,
        )
        payload = camera.tobytes()
        path = checkpoint_root / "selected_receiver_output" / f"batch_{start:04d}_{stop:04d}.raw"
        _publish_bytes(path, payload)
        row = {
            "pair_range": [start, stop],
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        rows.append(row)
        digest.update(payload)
        total += len(payload)
        del camera, payload
        gc.collect()
    if total != int(np.prod(CAMERA_SHAPE, dtype=np.int64)):
        raise PA2MeasurementError("selected receiver output byte count differs")
    return {
        "schema": "ddm_pa2_selected_receiver_output.v1",
        "stack": list(stack),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "stage_count": len(rows),
        "stages": rows,
        "all_stage_checkpoints_preserved": True,
        "assembly_order": "pair-major frame-major HWC uint8",
    }


def _baseline_from_settled_receipt(base: BaseConfig) -> dict[str, Any] | None:
    if base.kind != "raw":
        return None
    receipt = _read_json(_resolve(base.custody_receipt_path))
    endpoint = receipt.get("endpoint")
    if not isinstance(endpoint, dict):
        raise PA2MeasurementError("settled base receipt endpoint is absent")
    return {
        "stack": [],
        "errors": int(endpoint["errors"]),
        "sites": int(endpoint["sites"]),
        "d_seg": float(endpoint["d_seg"]),
        "d_pose": float(endpoint["d_pose"]),
        "advisory_objective_at_exact_archive_bytes": float(
            endpoint["advisory_objective_at_packed_bytes"]
        ),
        "archive_bytes": base.archive_bytes,
        "scorer_batch_size": 32,
        "stage_count": int(receipt["stage_count"]),
        "settled_receipt_reused": True,
    }


def _measure_base(
    *,
    config: DDMPA2ZeroByteDecodeFamilyConfigV1,
    base: BaseConfig,
    selected_steps: Sequence[int],
    labels: np.memmap,
    target_poses: np.memmap,
    segnet: Any,
    posenet: Any,
    bulk: Path,
) -> dict[str, Any]:
    checkpoint_root = bulk / base.base_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    loader = BaseLoader(base, selected_steps=selected_steps)
    pa1_affine = None
    candidates = list(STRICT_CANDIDATES)
    if base.kind == "ms2r":
        gain, bias, pa1_value = _pa1_affine(loader, checkpoint_root=checkpoint_root)
        pa1_affine = (gain, bias)
        candidates.insert(0, PA1_MEMBER)
    else:
        pa1_value = {
            "transform_id": PA1_MEMBER,
            "status": "ALREADY_IN_BASE_DO_NOT_REAPPLY",
            "base_id": base.base_id,
        }
    baseline = _baseline_from_settled_receipt(base)
    if baseline is None:
        baseline = _score_stack(
            loader=loader,
            base=base,
            stack=(),
            pa1_affine=pa1_affine,
            labels=labels,
            target_poses=target_poses,
            segnet=segnet,
            posenet=posenet,
            checkpoint_root=checkpoint_root,
        )
    blind = _blind_identity(
        loader=loader,
        base=base,
        checkpoint_root=checkpoint_root,
    )
    current = baseline
    current_stack: list[str] = []
    remaining = list(candidates)
    rounds: list[dict[str, Any]] = []
    while remaining:
        conditionals: list[dict[str, Any]] = []
        for member in remaining:
            result = _score_stack(
                loader=loader,
                base=base,
                stack=(*current_stack, member),
                pa1_affine=pa1_affine,
                labels=labels,
                target_poses=target_poses,
                segnet=segnet,
                posenet=posenet,
                checkpoint_root=checkpoint_root,
            )
            result["conditional_delta_objective"] = (
                result["advisory_objective_at_exact_archive_bytes"]
                - current["advisory_objective_at_exact_archive_bytes"]
            )
            result["conditional_delta_d_seg"] = result["d_seg"] - current["d_seg"]
            result["conditional_delta_d_pose"] = result["d_pose"] - current["d_pose"]
            result["conditional_delta_archive_bytes"] = 0
            conditionals.append(result)
        best = min(
            conditionals,
            key=lambda row: (
                row["advisory_objective_at_exact_archive_bytes"],
                row["stack"],
            ),
        )
        admitted = (
            best["advisory_objective_at_exact_archive_bytes"]
            < current["advisory_objective_at_exact_archive_bytes"]
        )
        rounds.append(
            {
                "round": len(rounds) + 1,
                "conditioned_on_stack": list(current_stack),
                "conditionals": conditionals,
                "selected": best["stack"][-1] if admitted else None,
                "strict_improvement": admitted,
            }
        )
        if not admitted:
            break
        current = best
        selected = current["stack"][-1]
        current_stack.append(selected)
        remaining.remove(selected)
    receiver_stack = [*current_stack, PA2Member.BLIND_ZERO_FILL.value]
    materialized = _materialize_selected(
        loader=loader,
        base=base,
        stack=receiver_stack,
        pa1_affine=pa1_affine,
        checkpoint_root=checkpoint_root,
    )
    archive_before = _sha256_file(Path(base.archive_path))
    archive_after = _sha256_file(Path(base.archive_path))
    if archive_before != archive_after:
        raise PA2MeasurementError("archive identity proof differs")
    if archive_after != (base.archive_bytes, base.archive_sha256):
        raise PA2MeasurementError("archive custody differs after PA2 receiver execution")
    value = {
        "schema": "ddm_pa2_zero_byte_decode_base_result.v1",
        "base_id": base.base_id,
        "baseline": baseline,
        "pa1_anchor": pa1_value,
        "blind_coordinate": blind,
        "greedy_non_telescoping_rounds": rounds,
        "strict_selected_stack": current_stack,
        "receiver_stack": receiver_stack,
        "joint_remeasure": current,
        "selected_receiver_output": materialized,
        "archive_identity": {
            "path": base.archive_path,
            "bytes_before": archive_before[0],
            "bytes_after": archive_after[0],
            "sha256_before": archive_before[1],
            "sha256_after": archive_after[1],
            "byte_delta": 0,
            "byte_identical": archive_before == archive_after,
        },
        "receiver_closure": {
            "base_custody_receipt": base.custody_receipt_path,
            "base_kind": (
                "real E4 inflate output plus generic PA2 interpreter"
                if base.kind == "raw"
                else "production q4/q8 quotient receiver plus generic PA2 interpreter"
            ),
            "generic_interpreter_source": (
                "src/tac/optimization/ddm_pa2_zero_byte_decode_family.py"
            ),
            "counted_video_derived_bytes_added": 0,
            "per_stage_receiver_output_preserved": True,
        },
        "evidence_axis": AXIS,
        "research_only": True,
        "score_claim": False,
    }
    _publish_json(checkpoint_root / "base_result.json", value)
    return value


def run(
    config_path: Path,
    *,
    only_base: str | None = None,
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(bulk)
    if usage.free < config.minimum_free_bytes:
        raise PA2MeasurementError(
            f"SSD preflight failed: {usage.free} < {config.minimum_free_bytes}"
        )
    scorer_path = _bound(
        config.scorer_config_path,
        config.scorer_config_sha256,
        "frozen scorer config",
    )
    scorer_config = FrozenScorerConfig.model_validate_json(
        scorer_path.read_bytes()
    )
    if scorer_config.pair_count != 600 or scorer_config.scorer_threads != 4:
        raise PA2MeasurementError("frozen scorer geometry or thread custody differs")
    target_path = _bound(
        scorer_config.target_cache_path,
        scorer_config.target_cache_sha256,
        "target cache",
    )
    labels = open_stored_npy_memmap(target_path, "lstars")
    target_poses = open_stored_npy_memmap(target_path, "gt_poses")
    if labels.shape != SEG_SHAPE or target_poses.shape != (600, 6):
        raise PA2MeasurementError("target cache geometry differs")
    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    segnet, posenet, scorer_custody = _load_models(scorer_config)
    scorer_custody["batch_size"] = 32
    ms2r_path = _bound(
        config.ms2r_receipt_path,
        config.ms2r_receipt_sha256,
        "MS2R receipt",
    )
    ms2r = _read_json(ms2r_path)
    selected_steps = ms2r["homotopy"]["solve"]["selected_steps"]
    results: list[dict[str, Any]] = []
    for base in config.bases:
        if only_base is not None and base.base_id != only_base:
            continue
        _bound(base.archive_path, base.archive_sha256, f"{base.base_id} archive")
        _bound(
            base.custody_receipt_path,
            base.custody_receipt_sha256,
            f"{base.base_id} custody receipt",
        )
        if base.kind == "raw":
            assert base.decoded_raw_path is not None
            raw = _bound(
                base.decoded_raw_path,
                str(base.decoded_raw_sha256),
                f"{base.base_id} decoded raw",
            )
            if raw.stat().st_size != int(np.prod(CAMERA_SHAPE, dtype=np.int64)):
                raise PA2MeasurementError("decoded raw geometry differs")
        results.append(
            _measure_base(
                config=config,
                base=base,
                selected_steps=selected_steps,
                labels=labels,
                target_poses=target_poses,
                segnet=segnet,
                posenet=posenet,
                bulk=bulk,
            )
        )
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "verdict": (
            "MEASURED_N600_BATCH32_ZERO_BYTE_FAMILY_CONDITIONALS; "
            "STRICT_GREEDY_STACKS_REMEASURED_JOINTLY"
        ),
        "verdict_scope": (
            "THREE INSTANCE BASES on macOS CPU frozen scorers; research-only "
            "advisory evidence, no contest-axis score or pointer mutation."
        ),
        "authority": {
            "evidence_axis": AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "pointer": config.pointer,
            "pointer_moved": False,
            "main_landing_review_required": True,
        },
        "typed_config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha256,
        },
        "storage": {
            "selected_tier": str(bulk),
            "free_bytes_at_preflight": usage.free,
            "minimum_free_bytes": config.minimum_free_bytes,
            "all_large_outputs_on_ssd": True,
            "cleanup_policy": (
                "certified immutable receiver stages are preserved on SSD; "
                "no local bulk and no destructive cleanup"
            ),
        },
        "family": family_inventory(),
        "pa1_anchor_id": PA1_MEMBER,
        "blocked_members": [row.to_dict() for row in BLOCKERS],
        "bases": results,
        "scorer_custody": scorer_custody,
        "batch_contract": {
            "pair_count": 600,
            "batch_size": 32,
            "stage_count_per_full_pass": 19,
            "non_telescoping": True,
            "greedy_joint_remeasure": True,
        },
        "directive_consumption": {
            "scorer_recursive_construction": (
                "#580 exact resize, exact frame ownership, stride-2 stem lattice, "
                "decoded-content temporal xi-hat"
            ),
            "free_interpreter_maximality": (
                "all decoded-frame derivations live in generic receiver code; "
                "no video-derived table was priced as code or added to archive"
            ),
        },
        "main_landing_review_required": True,
    }
    receipt_root = _resolve(config.receipt_root)
    receipt_path = (
        receipt_root / "receipt.json"
        if only_base is None
        else receipt_root / f"receipt_{only_base}.json"
    )
    _publish_json(receipt_path, receipt)
    return receipt


def aggregate_completed(config_path: Path) -> dict[str, Any]:
    """Build the repository receipt from already-complete SSD base receipts."""

    config, config_sha256 = _load_config(config_path)
    bulk = Path(config.bulk_root)
    results: list[dict[str, Any]] = []
    result_artifacts: list[dict[str, Any]] = []
    for base in config.bases:
        path = bulk / base.base_id / "base_result.json"
        if not path.is_file():
            raise PA2MeasurementError(f"base result is absent: {path}")
        value = _read_json(path)
        if (
            value.get("schema") != "ddm_pa2_zero_byte_decode_base_result.v1"
            or value.get("base_id") != base.base_id
            or value.get("archive_identity", {}).get("bytes_after") != base.archive_bytes
            or value.get("archive_identity", {}).get("sha256_after") != base.archive_sha256
            or value.get("archive_identity", {}).get("byte_identical") is not True
            or value.get("blind_coordinate", {}).get("n600_batch32_exact") is not True
            or value.get("selected_receiver_output", {}).get("stage_count") != 19
        ):
            raise PA2MeasurementError(f"base result contract differs: {path}")
        if _sha256_file(Path(base.archive_path)) != (
            base.archive_bytes,
            base.archive_sha256,
        ):
            raise PA2MeasurementError(f"archive drifted after base result: {base.base_id}")
        result_artifacts.append(
            {
                "base_id": base.base_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path)[1],
            }
        )
        results.append(value)
    summaries = [
        {
            "base_id": row["base_id"],
            "archive_bytes": row["archive_identity"]["bytes_after"],
            "archive_sha256": row["archive_identity"]["sha256_after"],
            "baseline_d_seg": row["baseline"]["d_seg"],
            "baseline_d_pose": row["baseline"]["d_pose"],
            "baseline_S": row["baseline"][
                "advisory_objective_at_exact_archive_bytes"
            ],
            "strict_selected_stack": row["strict_selected_stack"],
            "joint_d_seg": row["joint_remeasure"]["d_seg"],
            "joint_d_pose": row["joint_remeasure"]["d_pose"],
            "joint_S": row["joint_remeasure"][
                "advisory_objective_at_exact_archive_bytes"
            ],
            "delta_S": row["joint_remeasure"][
                "advisory_objective_at_exact_archive_bytes"
            ]
            - row["baseline"]["advisory_objective_at_exact_archive_bytes"],
            "archive_byte_delta": row["archive_identity"]["byte_delta"],
            "blind_coordinate_score_input_identity": row["blind_coordinate"][
                "n600_batch32_exact"
            ],
        }
        for row in results
    ]
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": config.lane_id,
        "verdict": (
            "IC2_FRAME1_XIHAT_ADMITTED_AT_ZERO_BYTES; "
            "IC1_AND_MS2R_RETAIN_SEALED_BASES"
        ),
        "verdict_scope": (
            "THREE INSTANCE BASES, n600 batch32 macOS CPU frozen scorers. "
            "The IC2 transform is research-only advisory and requires MAIN "
            "receiver integration/review before any contest-axis replay."
        ),
        "authority": {
            "evidence_axis": AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": config.pointer,
            "pointer_moved": False,
            "main_landing_review_required": True,
        },
        "typed_config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha256,
        },
        "storage": {
            "selected_tier": str(bulk),
            "all_large_outputs_on_ssd": True,
            "all_selected_outputs_stage_checkpointed": True,
            "cleanup_policy": (
                "certified immutable receiver stages remain on SSD; no local "
                "bulk and no destructive cleanup"
            ),
        },
        "family": family_inventory(),
        "pa1_anchor_id": PA1_MEMBER,
        "blocked_members": [row.to_dict() for row in BLOCKERS],
        "base_result_artifacts": result_artifacts,
        "base_summaries": summaries,
        "bases": results,
        "batch_contract": {
            "pair_count": 600,
            "batch_size": 32,
            "stage_count_per_full_pass": 19,
            "non_telescoping": True,
            "greedy_joint_remeasure": True,
        },
        "triality": {
            "dsl": (
                ".omx/research/configs/"
                "ddm_pa2_zero_byte_decode_family_20260724.json"
            ),
            "dag_feed": (
                ".omx/research/"
                "ddm_pa2_zero_byte_decode_family_DAG_FEED_20260724.md"
            ),
            "equation_id": "ddm_pa2_zero_byte_conditional_greedy_v1",
        },
        "directive_consumption": {
            "scorer_recursive_construction": (
                "#580 exact resize, exact frame ownership, stride-2 stem lattice, "
                "decoded-content temporal xi-hat"
            ),
            "free_interpreter_maximality": (
                "all decoded-frame derivations live in generic receiver code; "
                "no video-derived table was priced as code or added to archive"
            ),
        },
        "main_landing_review_required": True,
    }
    receipt_path = _resolve(config.receipt_root) / "receipt.json"
    _publish_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            REPO
            / ".omx/research/configs/ddm_pa2_zero_byte_decode_family_20260724.json"
        ),
    )
    parser.add_argument(
        "--base",
        choices=("IC1_W_joint_PA1", "IC2_W_seg_PA1", "MS2R_q4_q8"),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="merge three complete SSD base_result receipts without rerunning scorers",
    )
    args = parser.parse_args(argv)
    if args.aggregate_only and args.base is not None:
        parser.error("--aggregate-only and --base are mutually exclusive")
    result = (
        aggregate_completed(Path(args.config).resolve())
        if args.aggregate_only
        else run(Path(args.config).resolve(), only_base=args.base)
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "bases": [
                    {
                        "base_id": row["base_id"],
                        "strict_selected_stack": row["strict_selected_stack"],
                        "joint_remeasure": row["joint_remeasure"],
                    }
                    for row in result["bases"]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
