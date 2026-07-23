# SPDX-License-Identifier: MIT
"""Typed receiver-closed perturbation and scorer-sensitivity surface for DDM.

The API deliberately starts from the counted runtime members, reopens their
strict manifest, mutates exactly one description coordinate, realizes camera
RGB through the standalone receiver, and then measures the frozen scorer
outputs in canonical chunks.  It does not load scorer weights or choose a
candidate: callers supply the already-custodied scorer objects and targets.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import brotli
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from tac.optimization import ddm_runtime_receiver as runtime

PERTURBATION_SCHEMA = "DDMRuntimePerturbationV1"
SENSITIVITY_SCHEMA = "ddm_runtime_receiver_sensitivity.v1"
SCORE_BYTE_DUAL = 25.0 / 37_545_489.0


class RuntimeSensitivityError(ValueError):
    """A packet, perturbation, realization, or scorer contract failed closed."""


class DDMRuntimePerturbationV1(BaseModel):
    """One typed edit in a decoded, counted DDM stream."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMRuntimePerturbationV1"] = Field(
        default=PERTURBATION_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    stream: Literal[
        "base/chart.anchors",
        "base/chart.gradients",
        "base/chart.residuals",
        "semantic/composed",
    ]
    flat_index: StrictInt = Field(ge=0)
    delta: StrictInt = Field(ge=-255, le=255)
    expected_original_value: StrictInt | None = None
    pair_start: StrictInt = Field(ge=0, lt=600)
    pair_stop: StrictInt = Field(gt=0, le=600)
    scorer_batch_size: Literal[16] = 16
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMRuntimePerturbationV1:
        if self.delta == 0:
            raise ValueError("perturbation delta must be nonzero")
        if self.pair_stop <= self.pair_start:
            raise ValueError("pair_stop must exceed pair_start")
        return self


@dataclass(frozen=True)
class DDMRuntimeDecodedStateV1:
    """Strictly reopened runtime state used by perturbation consumers."""

    manifest: dict[str, Any]
    manifest_sha256: str
    anchors: torch.Tensor
    gradients: torch.Tensor
    residuals: torch.Tensor
    labels: torch.Tensor
    palette: torch.Tensor
    camera_rows: torch.Tensor
    camera_columns: torch.Tensor
    semantic_frame_policy: str
    chart_member: bytes
    semantic_member: bytes


@dataclass(frozen=True)
class DDMRuntimeRealizedPerturbationV1:
    """Baseline and one-coordinate-mutated receiver outputs."""

    perturbation: DDMRuntimePerturbationV1
    original_value: int
    perturbed_value: int
    baseline_camera: np.ndarray
    perturbed_camera: np.ndarray
    changed_camera_values: int
    baseline_camera_sha256: str
    perturbed_camera_sha256: str
    manifest_sha256: str
    member_name: str
    baseline_member_bytes: int
    perturbed_member_bytes: int
    baseline_member_sha256: str
    perturbed_member_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def decode_runtime_state(
    members: dict[str, bytes],
) -> DDMRuntimeDecodedStateV1:
    """Reopen exact packet members using the standalone receiver contract."""

    if tuple(members) != runtime.EXPECTED_MEMBERS:
        raise RuntimeSensitivityError("runtime members are incomplete or reordered")
    manifest_payload = members["manifest.json"]
    try:
        manifest = runtime._validate_manifest(
            runtime._duplicate_refusing_json(
                manifest_payload,
                label="manifest.json",
            ),
            members,
        )
        chart_raw, chart_shape = runtime._parse_blob(
            members["base/chart.ddb"],
            expected_kind=0,
            label="base/chart.ddb",
        )
        semantic_raw, semantic_shape = runtime._parse_blob(
            members["semantic/composed.dds"],
            expected_kind=1,
            label="semantic/composed.dds",
        )
    except runtime.ReceiverError as exc:
        raise RuntimeSensitivityError("runtime packet failed strict reopen") from exc
    if chart_shape or semantic_shape != (600, runtime.PAIR_H, runtime.PAIR_W):
        raise RuntimeSensitivityError("runtime stream geometry changed")
    anchors, gradients, residuals = runtime._chart_views(
        chart_raw,
        manifest["chart"],
    )
    labels = torch.frombuffer(bytearray(semantic_raw), dtype=torch.uint8).reshape(
        600,
        runtime.PAIR_H,
        runtime.PAIR_W,
    )
    palette = torch.tensor(
        manifest["output"]["palette_rgb_u8"],
        dtype=torch.uint8,
    )
    if int(labels.max()) >= len(palette):
        raise RuntimeSensitivityError("semantic labels escape the counted palette")
    camera_rows = torch.div(
        torch.arange(runtime.CAMERA_H, dtype=torch.int64) * runtime.PAIR_H,
        runtime.CAMERA_H,
        rounding_mode="floor",
    )
    camera_columns = torch.div(
        torch.arange(runtime.CAMERA_W, dtype=torch.int64) * runtime.PAIR_W,
        runtime.CAMERA_W,
        rounding_mode="floor",
    )
    return DDMRuntimeDecodedStateV1(
        manifest=manifest,
        manifest_sha256=_sha256(manifest_payload),
        anchors=anchors,
        gradients=gradients,
        residuals=residuals,
        labels=labels,
        palette=palette,
        camera_rows=camera_rows,
        camera_columns=camera_columns,
        semantic_frame_policy=str(
            manifest.get("semantic_frame_policy", "both_frames")
        ),
        chart_member=members["base/chart.ddb"],
        semantic_member=members["semantic/composed.dds"],
    )


def _edited_tensor(
    state: DDMRuntimeDecodedStateV1,
    perturbation: DDMRuntimePerturbationV1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    tensors = {
        "base/chart.anchors": state.anchors,
        "base/chart.gradients": state.gradients,
        "base/chart.residuals": state.residuals,
        "semantic/composed": state.labels,
    }
    source = tensors[perturbation.stream]
    if perturbation.flat_index >= source.numel():
        raise RuntimeSensitivityError("perturbation coordinate escapes its stream")
    edited = source.clone()
    flat = edited.reshape(-1)
    original = int(flat[perturbation.flat_index])
    if (
        perturbation.expected_original_value is not None
        and perturbation.expected_original_value != original
    ):
        raise RuntimeSensitivityError("perturbation original-value precondition failed")
    changed = original + perturbation.delta
    if perturbation.stream == "semantic/composed":
        if not 0 <= changed < len(state.palette):
            raise RuntimeSensitivityError("semantic perturbation escapes the palette")
    elif not -32768 <= changed <= 32767:
        raise RuntimeSensitivityError("chart perturbation escapes int16")
    flat[perturbation.flat_index] = changed
    return (
        edited if perturbation.stream == "base/chart.anchors" else state.anchors,
        edited if perturbation.stream == "base/chart.gradients" else state.gradients,
        edited if perturbation.stream == "base/chart.residuals" else state.residuals,
        edited if perturbation.stream == "semantic/composed" else state.labels,
        original,
        changed,
    )


def _render(
    state: DDMRuntimeDecodedStateV1,
    *,
    start: int,
    stop: int,
    anchors: torch.Tensor,
    gradients: torch.Tensor,
    residuals: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    try:
        camera = runtime._render_batch(
            start=start,
            stop=stop,
            anchors=anchors,
            gradients=gradients,
            residuals=residuals,
            labels=labels,
            palette=state.palette,
            camera_rows=state.camera_rows,
            camera_columns=state.camera_columns,
            semantic_frame_policy=state.semantic_frame_policy,
        )
    except runtime.ReceiverError as exc:
        raise RuntimeSensitivityError("perturbed state failed receiver realization") from exc
    return np.ascontiguousarray(camera.numpy())


def _pack_blob(raw: bytes, *, kind: int, dimensions: tuple[int, ...]) -> bytes:
    """Re-encode one edited counted stream with the runtime blob contract."""

    if kind not in {0, 1}:
        raise RuntimeSensitivityError("runtime blob kind is invalid")
    if kind == 0 and dimensions:
        raise RuntimeSensitivityError("opaque runtime blob must be rank zero")
    product = math.prod(dimensions)
    if kind == 1 and product != len(raw):
        raise RuntimeSensitivityError("edited stream dimensions disagree with bytes")
    coded = brotli.compress(raw, quality=11)
    return (
        runtime.BLOB_HEADER.pack(
            runtime.BLOB_MAGIC,
            1,
            1,
            kind,
            len(dimensions),
            len(raw),
            len(coded),
            hashlib.sha256(raw).digest(),
        )
        + (
            struct.pack(f">{len(dimensions)}I", *dimensions)
            if dimensions
            else b""
        )
        + coded
    )


def _serialized_edit(
    state: DDMRuntimeDecodedStateV1,
    perturbation: DDMRuntimePerturbationV1,
    *,
    anchors: torch.Tensor,
    gradients: torch.Tensor,
    residuals: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[str, bytes, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serialize, parse back, and return the one edited counted member."""

    if perturbation.stream == "semantic/composed":
        member_name = "semantic/composed.dds"
        raw = np.asarray(labels.numpy(), dtype=np.uint8).tobytes(order="C")
        dimensions = tuple(int(value) for value in labels.shape)
        member = _pack_blob(
            raw,
            kind=1,
            dimensions=dimensions,
        )
        parsed_raw, parsed_shape = runtime._parse_blob(
            member,
            expected_kind=1,
            label=member_name,
        )
        if parsed_shape != dimensions:
            raise RuntimeSensitivityError("edited semantic member changed geometry")
        parsed_labels = torch.frombuffer(
            bytearray(parsed_raw),
            dtype=torch.uint8,
        ).reshape(dimensions)
        return (
            member_name,
            member,
            len(state.semantic_member),
            anchors,
            gradients,
            residuals,
            parsed_labels,
        )
    member_name = "base/chart.ddb"
    raw = b"".join(
        np.asarray(value.numpy(), dtype="<i2").tobytes(order="C")
        for value in (anchors, gradients, residuals)
    )
    member = _pack_blob(raw, kind=0, dimensions=())
    parsed_raw, parsed_shape = runtime._parse_blob(
        member,
        expected_kind=0,
        label=member_name,
    )
    if parsed_shape:
        raise RuntimeSensitivityError("edited chart member changed geometry")
    parsed_anchors, parsed_gradients, parsed_residuals = runtime._chart_views(
        parsed_raw,
        state.manifest["chart"],
    )
    return (
        member_name,
        member,
        len(state.chart_member),
        parsed_anchors,
        parsed_gradients,
        parsed_residuals,
        labels,
    )


def realize_perturbation(
    state: DDMRuntimeDecodedStateV1,
    perturbation: DDMRuntimePerturbationV1,
) -> DDMRuntimeRealizedPerturbationV1:
    """Realize the base and one-coordinate edit through the exact receiver."""

    anchors, gradients, residuals, labels, original, changed = _edited_tensor(
        state,
        perturbation,
    )
    (
        member_name,
        perturbed_member,
        baseline_member_bytes,
        anchors,
        gradients,
        residuals,
        labels,
    ) = _serialized_edit(
        state,
        perturbation,
        anchors=anchors,
        gradients=gradients,
        residuals=residuals,
        labels=labels,
    )
    baseline = _render(
        state,
        start=perturbation.pair_start,
        stop=perturbation.pair_stop,
        anchors=state.anchors,
        gradients=state.gradients,
        residuals=state.residuals,
        labels=state.labels,
    )
    perturbed = _render(
        state,
        start=perturbation.pair_start,
        stop=perturbation.pair_stop,
        anchors=anchors,
        gradients=gradients,
        residuals=residuals,
        labels=labels,
    )
    changed_values = int(np.count_nonzero(baseline != perturbed))
    if changed_values <= 0:
        raise RuntimeSensitivityError(
            "counted-to-output #417 check failed: edit is receiver-inert"
        )
    return DDMRuntimeRealizedPerturbationV1(
        perturbation=perturbation,
        original_value=original,
        perturbed_value=changed,
        baseline_camera=baseline,
        perturbed_camera=perturbed,
        changed_camera_values=changed_values,
        baseline_camera_sha256=_sha256(baseline.tobytes(order="C")),
        perturbed_camera_sha256=_sha256(perturbed.tobytes(order="C")),
        manifest_sha256=state.manifest_sha256,
        member_name=member_name,
        baseline_member_bytes=baseline_member_bytes,
        perturbed_member_bytes=len(perturbed_member),
        baseline_member_sha256=_sha256(
            state.semantic_member
            if member_name == "semantic/composed.dds"
            else state.chart_member
        ),
        perturbed_member_sha256=_sha256(perturbed_member),
    )


def _forward(
    segnet: Any,
    posenet: Any,
    camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(camera)
    if (
        value.dtype != np.uint8
        or value.ndim != 5
        or value.shape[1:] != (2, runtime.CAMERA_H, runtime.CAMERA_W, 3)
    ):
        raise RuntimeSensitivityError("scorer input has wrong pair geometry")
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
        output = posenet(posenet.preprocess_input(tensor))
        pose = output["pose"] if isinstance(output, dict) else output
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(cells), np.ascontiguousarray(pose6)


def _distortions(
    *,
    cells: np.ndarray,
    pose6: np.ndarray,
    target_labels: np.ndarray,
    target_poses: np.ndarray,
) -> tuple[float, float]:
    if (
        target_labels.shape != cells.shape
        or target_poses.shape != pose6.shape
        or target_labels.dtype != np.uint8
        or target_poses.dtype != np.float64
    ):
        raise RuntimeSensitivityError("target scorer custody has wrong shape or dtype")
    return (
        float(np.count_nonzero(cells != target_labels) / cells.size),
        float(np.square(pose6 - target_poses).mean(dtype=np.float64)),
    )


def score_realized_perturbation(
    realized: DDMRuntimeRealizedPerturbationV1,
    *,
    segnet: Any,
    posenet: Any,
    target_labels: np.ndarray,
    target_poses: np.ndarray,
) -> dict[str, Any]:
    """Measure one realized edit with canonical batch-16 scorer geometry."""

    pair_count = realized.perturbation.pair_stop - realized.perturbation.pair_start
    if (
        target_labels.shape != (pair_count, runtime.PAIR_H, runtime.PAIR_W)
        or target_poses.shape != (pair_count, 6)
    ):
        raise RuntimeSensitivityError("targets do not cover the perturbation window")
    baseline_cells: list[np.ndarray] = []
    baseline_pose: list[np.ndarray] = []
    perturbed_cells: list[np.ndarray] = []
    perturbed_pose: list[np.ndarray] = []
    for start in range(0, pair_count, realized.perturbation.scorer_batch_size):
        stop = min(start + realized.perturbation.scorer_batch_size, pair_count)
        cells0, pose0 = _forward(
            segnet,
            posenet,
            realized.baseline_camera[start:stop],
        )
        cells1, pose1 = _forward(
            segnet,
            posenet,
            realized.perturbed_camera[start:stop],
        )
        baseline_cells.append(cells0)
        baseline_pose.append(pose0)
        perturbed_cells.append(cells1)
        perturbed_pose.append(pose1)
    cells0 = np.concatenate(baseline_cells)
    pose0 = np.concatenate(baseline_pose)
    cells1 = np.concatenate(perturbed_cells)
    pose1 = np.concatenate(perturbed_pose)
    dseg0, dpose0 = _distortions(
        cells=cells0,
        pose6=pose0,
        target_labels=target_labels,
        target_poses=target_poses,
    )
    dseg1, dpose1 = _distortions(
        cells=cells1,
        pose6=pose1,
        target_labels=target_labels,
        target_poses=target_poses,
    )
    score0 = 100.0 * dseg0 + math.sqrt(10.0 * dpose0)
    score1 = 100.0 * dseg1 + math.sqrt(10.0 * dpose1)
    delta_bytes = (
        realized.perturbed_member_bytes - realized.baseline_member_bytes
    )
    delta_score = score1 - score0 + SCORE_BYTE_DUAL * delta_bytes
    return {
        "schema": SENSITIVITY_SCHEMA,
        "first_rung": True,
        "perturbation": realized.perturbation.model_dump(
            mode="json",
            by_alias=True,
        ),
        "manifest_sha256": realized.manifest_sha256,
        "receiver_bijection": {
            "counted_to_output_changed": True,
            "changed_camera_values": realized.changed_camera_values,
            "baseline_camera_sha256": realized.baseline_camera_sha256,
            "perturbed_camera_sha256": realized.perturbed_camera_sha256,
            "output_to_single_owner": realized.perturbation.stream,
            "serialized_member": realized.member_name,
            "baseline_member_sha256": realized.baseline_member_sha256,
            "perturbed_member_sha256": realized.perturbed_member_sha256,
        },
        "baseline": {
            "d_seg": dseg0,
            "d_pose": dpose0,
            "member_bytes": realized.baseline_member_bytes,
        },
        "perturbed": {
            "d_seg": dseg1,
            "d_pose": dpose1,
            "member_bytes": realized.perturbed_member_bytes,
        },
        "delta": {
            "d_seg": dseg1 - dseg0,
            "d_pose": dpose1 - dpose0,
            "bytes": delta_bytes,
            "score": delta_score,
            "score_per_byte": None if delta_bytes == 0 else delta_score / delta_bytes,
            "break_even_score_per_byte": SCORE_BYTE_DUAL,
        },
        "scorer": {
            "batch_size": 16,
            "batch_count": math.ceil(pair_count / 16),
            "segnet_argmax": True,
            "posenet_official_two_frame_path": True,
        },
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "score_claim": False,
        "verdict_scope": (
            "One typed receiver coordinate and one bounded pair window; "
            "this row ranks the edit, not its whole stream family."
        ),
    }
