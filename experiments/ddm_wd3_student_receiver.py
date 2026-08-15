#!/usr/bin/env python3
"""Adaptive counted packet and receiver binding for WD3 width distillation.

WD3 keeps the WD2 student topology but replaces WD2's uniform int4 packet with
an explicit per-group 2--8 bit allocation.  The same allocation drives the
training-time STE and the receiver parser.  Learned bytes stay in archive.zip.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from experiments import ddm_wd2_student_receiver as wd2
except ImportError:  # retained cpr1 runtime imports this file beside wd2_receiver.py
    import wd2_receiver as wd2  # type: ignore[no-redef]

MAGIC = b"WD3Q"
VERSION = 1
HEADER = struct.Struct("<4sBBBBBII")
ALLOCATION_SCHEMA = "ddm_wd3_adaptive_quant_allocation.v1"
PACKET_SCHEMA = "ddm_wd3_adaptive_student_packet.v1"

StudentSpec = wd2.StudentSpec
StudentSemanticRenderer = wd2.StudentSemanticRenderer
N = wd2.N
EVAL_H = wd2.EVAL_H
EVAL_W = wd2.EVAL_W
CAMERA_H = wd2.CAMERA_H
CAMERA_W = wd2.CAMERA_W


class WD3ReceiverError(RuntimeError):
    """The adaptive packet or retained receiver binding failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _quant_axis(name: str, value: torch.Tensor) -> int | None:
    if value.ndim < 2:
        return None
    return value.ndim - 1 if name.endswith("embed.weight") else 0


def _group_count(name: str, value: torch.Tensor) -> int:
    axis = _quant_axis(name, value)
    return 0 if axis is None else int(value.shape[axis])


@dataclass(frozen=True)
class AdaptiveQuantizationAllocation:
    """Canonical bit depth for every packet-quantized parameter group."""

    bits: Mapping[str, tuple[int, ...]]
    selection_sha256: str
    policy: str = "uniform_int4_degenerate"

    def validate(self, model: nn.Module) -> None:
        expected = {name: _group_count(name, value) for name, value in model.state_dict().items() if value.ndim >= 2}
        if set(self.bits) != set(expected):
            missing = sorted(set(expected) - set(self.bits))
            extra = sorted(set(self.bits) - set(expected))
            raise WD3ReceiverError(f"allocation tensor set differs; missing={missing}, extra={extra}")
        for name, count in expected.items():
            depths = self.bits[name]
            if len(depths) != count:
                raise WD3ReceiverError(f"allocation group count differs for {name}")
            if any(type(bit) is not int or not 2 <= bit <= 8 for bit in depths):
                raise WD3ReceiverError(f"allocation bit depth is outside [2,8] for {name}")
        if len(self.selection_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.selection_sha256):
            raise WD3ReceiverError("allocation selection SHA-256 is not canonical")
        if not self.policy:
            raise WD3ReceiverError("allocation policy is empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ALLOCATION_SCHEMA,
            "policy": self.policy,
            "selection_sha256": self.selection_sha256,
            "bits": {name: list(self.bits[name]) for name in sorted(self.bits)},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptiveQuantizationAllocation:
        if set(value) != {"schema", "policy", "selection_sha256", "bits"}:
            raise WD3ReceiverError("allocation fields differ from the WD3 schema")
        if value.get("schema") != ALLOCATION_SCHEMA or not isinstance(value.get("bits"), dict):
            raise WD3ReceiverError("allocation schema differs")
        return cls(
            bits={str(name): tuple(depths) for name, depths in value["bits"].items()},
            selection_sha256=str(value["selection_sha256"]),
            policy=str(value["policy"]),
        )


def uniform_allocation(
    model: nn.Module,
    bits: int = 4,
    *,
    selection_sha256: str = "0" * 64,
) -> AdaptiveQuantizationAllocation:
    if not 2 <= bits <= 8:
        raise WD3ReceiverError("uniform bit depth must be in [2,8]")
    allocation = AdaptiveQuantizationAllocation(
        bits={
            name: (bits,) * _group_count(name, value) for name, value in model.state_dict().items() if value.ndim >= 2
        },
        selection_sha256=selection_sha256,
        policy=f"uniform_int{bits}_degenerate",
    )
    allocation.validate(model)
    return allocation


def _signed_limits(bits: int) -> tuple[int, int]:
    maximum = (1 << (bits - 1)) - 1
    return -maximum, maximum


def _pack_signed(values: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(values, dtype=np.int16).reshape(-1)
    minimum, maximum = _signed_limits(bits)
    if np.any(flat < minimum) or np.any(flat > maximum):
        raise WD3ReceiverError(f"signed int{bits} values exceed the canonical range")
    mask = (1 << bits) - 1
    unsigned = (flat.astype(np.int64) & mask).astype(np.uint16)
    bit_count = int(unsigned.size) * bits
    packed = np.zeros((bit_count + 7) // 8, dtype=np.uint8)
    cursor = 0
    for value in unsigned:
        number = int(value)
        for shift in range(bits):
            if number & (1 << shift):
                packed[cursor >> 3] |= np.uint8(1 << (cursor & 7))
            cursor += 1
    return packed.tobytes()


def _unpack_signed(blob: memoryview, count: int, bits: int) -> tuple[np.ndarray, memoryview]:
    byte_count = (count * bits + 7) // 8
    if len(blob) < byte_count:
        raise WD3ReceiverError(f"truncated signed int{bits} group")
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    unsigned = np.zeros(count, dtype=np.int16)
    cursor = 0
    for index in range(count):
        number = 0
        for shift in range(bits):
            if int(packed[cursor >> 3]) & (1 << (cursor & 7)):
                number |= 1 << shift
            cursor += 1
        unsigned[index] = number
    sign = 1 << (bits - 1)
    signed = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned).astype(np.int16, copy=False)
    minimum, _ = _signed_limits(bits)
    if np.any(signed < minimum):
        raise WD3ReceiverError(f"reserved most-negative int{bits} code")
    return signed, blob[byte_count:]


def _groups(value: torch.Tensor, axis: int) -> torch.Tensor:
    return value.movedim(axis, 0).contiguous()


def quantize_tensor_groups(value: torch.Tensor, *, axis: int, bits: Sequence[int]) -> torch.Tensor:
    """Quantize each group on its own fp16 scale and signed grid."""

    source = value.detach().to(device="cpu", dtype=torch.float32)
    grouped = _groups(source, axis)
    if grouped.shape[0] != len(bits):
        raise WD3ReceiverError("tensor group count differs from allocation")
    restored = torch.empty_like(grouped)
    for index, depth in enumerate(bits):
        _, maximum = _signed_limits(depth)
        maximum_abs = grouped[index].abs().amax()
        scale = (
            torch.where(
                maximum_abs > 0,
                maximum_abs / float(maximum),
                torch.ones_like(maximum_abs),
            )
            .to(torch.float16)
            .to(torch.float32)
        )
        codes = torch.round(grouped[index] / scale).clamp(-maximum, maximum)
        restored[index] = codes * scale
    return restored.movedim(0, axis)


def fake_quantize_state(model: nn.Module, allocation: AdaptiveQuantizationAllocation) -> dict[str, torch.Tensor]:
    """Return STE parameters whose forward values equal WD3 packet parse-back."""

    allocation.validate(model)
    result: dict[str, torch.Tensor] = {}
    parameters = dict(model.named_parameters())
    for name, value in model.state_dict().items():
        if value.ndim < 2:
            quantized = value.detach().cpu().to(torch.float16).to(torch.float32)
        else:
            axis = _quant_axis(name, value)
            assert axis is not None
            quantized = quantize_tensor_groups(value, axis=axis, bits=allocation.bits[name])
        quantized = quantized.to(device=value.device, dtype=value.dtype)
        if name in parameters:
            live = parameters[name]
            result[name] = live + (quantized - live).detach()
        else:
            result[name] = quantized
    return result


def _packet_metadata(
    model: StudentSemanticRenderer,
    allocation: AdaptiveQuantizationAllocation,
) -> dict[str, Any]:
    return {
        "schema": PACKET_SCHEMA,
        "spec": model.spec.as_dict(),
        "allocation": allocation.as_dict(),
    }


def pack_student(
    model: StudentSemanticRenderer,
    allocation: AdaptiveQuantizationAllocation,
) -> bytes:
    """Serialize the actual adaptive deployment state."""

    allocation.validate(model)
    body = bytearray()
    for name, value in model.state_dict().items():
        source = value.detach().cpu().float()
        axis = _quant_axis(name, source)
        if axis is None:
            body.extend(source.numpy().astype("<f2", copy=False).tobytes())
            continue
        grouped = _groups(source, axis)
        for index, depth in enumerate(allocation.bits[name]):
            _, maximum = _signed_limits(depth)
            maximum_abs = grouped[index].abs().amax()
            scale = torch.where(
                maximum_abs > 0,
                maximum_abs / float(maximum),
                torch.ones_like(maximum_abs),
            )
            scale16 = np.asarray(float(scale), dtype="<f2")
            restored_scale = float(scale16.astype(np.float32))
            codes = torch.round(grouped[index] / restored_scale).clamp(-maximum, maximum)
            body.extend(scale16.tobytes())
            body.extend(_pack_signed(codes.to(torch.int16).numpy(), depth))
    metadata = canonical_json_bytes(_packet_metadata(model, allocation))
    spec = model.spec
    return (
        HEADER.pack(
            MAGIC,
            VERSION,
            wd2.FORM_TO_ID[spec.form],
            spec.width,
            spec.depth,
            spec.rank,
            len(metadata),
            len(body),
        )
        + metadata
        + bytes(body)
    )


def packet_metadata(blob: bytes) -> dict[str, Any]:
    if len(blob) < HEADER.size:
        raise WD3ReceiverError("truncated WD3 header")
    magic, version, form_id, width, depth, rank, meta_bytes, body_bytes = HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION or form_id not in wd2.ID_TO_FORM:
        raise WD3ReceiverError("unsupported WD3 header")
    if len(blob) != HEADER.size + meta_bytes + body_bytes:
        raise WD3ReceiverError("WD3 packet length differs")
    try:
        metadata = json.loads(blob[HEADER.size : HEADER.size + meta_bytes])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WD3ReceiverError("WD3 metadata is not canonical JSON") from error
    if canonical_json_bytes(metadata) != blob[HEADER.size : HEADER.size + meta_bytes]:
        raise WD3ReceiverError("WD3 metadata JSON is noncanonical")
    if set(metadata) != {"schema", "spec", "allocation"}:
        raise WD3ReceiverError("WD3 metadata fields differ")
    if metadata.get("schema") != PACKET_SCHEMA:
        raise WD3ReceiverError("WD3 packet schema differs")
    expected = {
        "candidate_id": metadata["spec"].get("candidate_id"),
        "form": wd2.ID_TO_FORM[form_id],
        "width": width,
        "depth": depth,
        "rank": rank,
    }
    if metadata["spec"] != expected:
        raise WD3ReceiverError("WD3 header and topology metadata differ")
    return metadata


def unpack_student(
    blob: bytes,
) -> StudentSemanticRenderer:
    metadata = packet_metadata(blob)
    spec = StudentSpec(**metadata["spec"])
    model = StudentSemanticRenderer(spec)
    allocation = AdaptiveQuantizationAllocation.from_dict(metadata["allocation"])
    allocation.validate(model)
    meta_bytes = HEADER.unpack_from(blob)[6]
    remaining = memoryview(blob)[HEADER.size + meta_bytes :]
    restored: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        shape = tuple(value.shape)
        axis = _quant_axis(name, value)
        if axis is None:
            byte_count = value.numel() * 2
            if len(remaining) < byte_count:
                raise WD3ReceiverError(f"truncated WD3 fp16 tensor {name}")
            array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            restored[name] = torch.from_numpy(array).reshape(shape).float()
            remaining = remaining[byte_count:]
            continue
        grouped_shape = tuple(value.movedim(axis, 0).shape)
        group_elements = int(np.prod(grouped_shape[1:], dtype=np.int64))
        groups = []
        for depth in allocation.bits[name]:
            if len(remaining) < 2:
                raise WD3ReceiverError(f"truncated WD3 scale for {name}")
            scale = float(np.frombuffer(remaining[:2], dtype="<f2")[0])
            remaining = remaining[2:]
            if not np.isfinite(scale) or scale <= 0:
                raise WD3ReceiverError(f"invalid WD3 scale for {name}")
            codes, remaining = _unpack_signed(remaining, group_elements, depth)
            groups.append(torch.from_numpy(codes.copy()).float().reshape(grouped_shape[1:]) * scale)
        restored[name] = torch.stack(groups).movedim(0, axis)
    if remaining:
        raise WD3ReceiverError(f"WD3 packet has {len(remaining)} trailing bytes")
    model.load_state_dict(restored, strict=True)
    return model


def packet_allocation(blob: bytes) -> AdaptiveQuantizationAllocation:
    metadata = packet_metadata(blob)
    allocation = AdaptiveQuantizationAllocation.from_dict(metadata["allocation"])
    model = StudentSemanticRenderer(StudentSpec(**metadata["spec"]))
    allocation.validate(model)
    return allocation


def serialized_bytes_for_allocation(
    model: StudentSemanticRenderer,
    allocation: AdaptiveQuantizationAllocation,
) -> int:
    """Exact packet size for this topology/allocation, independent of values."""

    allocation.validate(model)
    metadata_bytes = len(canonical_json_bytes(_packet_metadata(model, allocation)))
    body_bytes = 0
    for name, value in model.state_dict().items():
        axis = _quant_axis(name, value)
        if axis is None:
            body_bytes += value.numel() * 2
            continue
        group_elements = value.numel() // value.shape[axis]
        body_bytes += sum(2 + (group_elements * bit + 7) // 8 for bit in allocation.bits[name])
    return HEADER.size + metadata_bytes + body_bytes


def allocation_telemetry(
    model: StudentSemanticRenderer,
    allocation: AdaptiveQuantizationAllocation,
) -> dict[str, Any]:
    allocation.validate(model)
    rows = []
    histogram = {str(bit): 0 for bit in range(2, 9)}
    for name, value in model.state_dict().items():
        axis = _quant_axis(name, value)
        if axis is None:
            continue
        group_elements = value.numel() // value.shape[axis]
        depths = allocation.bits[name]
        for bit in depths:
            histogram[str(bit)] += 1
        rows.append(
            {
                "tensor": name,
                "axis": axis,
                "groups": len(depths),
                "group_elements": group_elements,
                "bits": list(depths),
                "coded_bytes_without_metadata": sum(2 + (group_elements * bit + 7) // 8 for bit in depths),
            }
        )
    return {
        "schema": "ddm_wd3_adaptive_quant_telemetry.v1",
        "selection_sha256": allocation.selection_sha256,
        "policy": allocation.policy,
        "bit_depth_group_histogram": histogram,
        "packet_bytes": serialized_bytes_for_allocation(model, allocation),
        "tensors": rows,
    }


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise WD3ReceiverError(f"runtime patch point differs: {label}")
    return source.replace(old, new)


def patch_runtime_tree(source_tree: Path, destination: Path) -> dict[str, Any]:
    """Retain WD2 behavior and add a narrowly dispatched WD3Q receiver branch."""

    wd2_receipt = wd2.patch_runtime_tree(source_tree, destination)
    residual_path = destination / "runtime/residual_archive.py"
    residual = residual_path.read_text(encoding="utf-8")
    residual = _replace_once(
        residual,
        'if semantic_body.startswith(b"WD2S"):',
        'if semantic_body.startswith((b"WD2S", b"WD3Q")):',
        "residual WD3 dispatch",
    )
    residual_path.write_text(residual, encoding="utf-8")

    f26_path = destination / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    f26 = _replace_once(
        f26,
        'wd2_student = parts.semantic_blob.startswith(b"WD2S")',
        'wd2_student = parts.semantic_blob.startswith((b"WD2S", b"WD3Q"))',
        "F26 WD3 semantic guard",
    )
    f26 = _replace_once(
        f26,
        'receiver_path = renderer_dir / "wd2_receiver.py"',
        'receiver_path = renderer_dir / ("wd3_receiver.py" if parts.semantic_blob.startswith(b"WD3Q") else "wd2_receiver.py")',
        "F26 WD3 receiver selection",
    )
    f26_path.write_text(f26, encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), destination / "cpr1/wd3_receiver.py")
    files = [
        {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return {
        "schema": "ddm_wd3_runtime_patch.v1",
        "source_tree": str(source_tree.resolve()),
        "destination": str(destination.resolve()),
        "student_magic": MAGIC.decode("ascii"),
        "wd2_receipt": wd2_receipt,
        "inactive_and_wd2_branches_retained": True,
        "files": files,
    }


def bind_archive(runtime_tree: Path, archive_path: Path) -> dict[str, Any]:
    return wd2.bind_archive(runtime_tree, archive_path)


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "EVAL_H",
    "EVAL_W",
    "HEADER",
    "MAGIC",
    "AdaptiveQuantizationAllocation",
    "N",
    "StudentSemanticRenderer",
    "StudentSpec",
    "WD3ReceiverError",
    "allocation_telemetry",
    "bind_archive",
    "fake_quantize_state",
    "pack_student",
    "packet_allocation",
    "packet_metadata",
    "patch_runtime_tree",
    "quantize_tensor_groups",
    "serialized_bytes_for_allocation",
    "uniform_allocation",
    "unpack_student",
]
