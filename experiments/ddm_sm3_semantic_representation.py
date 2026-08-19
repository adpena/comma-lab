#!/usr/bin/env python3
"""Materialize retained PR130 semantic-representation candidates.

This is the scorer-free half of DDM-SM3.  It decomposes the exact shipped
semantic payload by tensor, then builds complete archives for several smaller
representations.  Every encoded payload is retained before the run exits.

The ``SM3R`` formats are checkpoint-template-bound research formats.  They are
not understood by the public receiver and they carry no score claim.  Their
purpose is to put real archive bytes, exact parse-back, and a deterministic
decoder object behind the candidates that a scorer owner can subsequently
measure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sd1 = importlib.import_module("experiments.ddm_sd1_semantic_rd_curve")


SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_BASE_ARCHIVE = SSD_ROOT / "ddm_pr130_reproduce_20260809/reproduction/archive.zip"
DEFAULT_CHECKPOINT = (
    SSD_ROOT
    / "pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints"
    / "semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)
DEFAULT_OUT = SSD_ROOT / "ddm_sm3_20260810"
EXPECTED_CHECKPOINT_SHA256 = sd1.EXPECTED_CHECKPOINT_SHA256
EXPECTED_BASE_ARCHIVE_SHA256 = sd1.EXPECTED_BASE_ARCHIVE_SHA256
EXPECTED_BASE_ARCHIVE_BYTES = sd1.EXPECTED_BASE_ARCHIVE_BYTES
ORIGINAL_BYTES = sd1.ORIGINAL_BYTES

MAGIC = b"SM3R"
VERSION = 1
MODE_VECTOR_VQ = 1
MODE_SCALE_VQ = 2
MODE_BOTH_VQ = 3
MODE_LOWRANK = 4
MODE_ROW_PRUNE = 5
MODE_ROW_PRUNE_MIXED = 6

LOWRANK_NAMES = {
    "coord_mix.weight",
    "blocks.0.pw.weight",
    "blocks.1.pw.weight",
    "blocks.2.pw.weight",
    "blocks.3.pw.weight",
}
PRUNE_NAMES = {
    "blocks.1.film.weight",
    "blocks.2.film.weight",
    "blocks.3.film.weight",
}
SELECTED_MIXED_Q3_NAMES = {
    "frame_embed.weight",
    "blocks.1.film.weight",
    "blocks.2.film.weight",
    "blocks.3.film.weight",
}
# The two SD1M "S2" legs that do NOT overlap the row-prune selection.  Films
# 1/2/3 are 99%-pruned by MODE_ROW_PRUNE at keep_percent=1, so re-quantizing
# them buys almost nothing; frame_embed and blocks.0.film are untouched by the
# prune and keep their full q4 payload, so they are the surviving marginal.
PRUNE_MIXED_Q3_NAMES = {
    "frame_embed.weight",
    "blocks.0.film.weight",
}
DEFAULT_SEMANTIC_BITS = 4
SD1_PUBLIC_RECEIVER_COMMIT = "58f62cd22ff07562c0534c999d705fb9edfe5279"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def retain_payload(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"resume artifact differs: {path}")
    atomic_write_bytes(path, payload)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"retained artifact byte count differs: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"retained artifact SHA-256 differs: {path}")


def load_retained_candidate(candidate_id: str, out_dir: Path) -> dict[str, Any]:
    receipt_path = out_dir / "retained/candidates" / candidate_id / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("candidate_id") != candidate_id:
        raise ValueError(f"resume candidate identity differs: {candidate_id}")
    for record in receipt["retained"].values():
        verify_artifact(record)
    receipt["receipt"] = artifact(receipt_path)
    return receipt


def require_ssd(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SSD_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"DDM-SM3 evidence must live below {SSD_ROOT}: {resolved}") from error
    return resolved


def pack_unsigned_bits(values: np.ndarray, bits: int) -> bytes:
    if not 1 <= bits <= 8:
        raise ValueError("unsigned bit width must be in [1, 8]")
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    if flat.size and int(flat.max()) >= (1 << bits):
        raise ValueError("unsigned value exceeds bit width")
    shifts = np.arange(bits, dtype=np.uint16)
    stream = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(stream, bitorder="little").tobytes()


def unpack_unsigned_bits(blob: memoryview, count: int, bits: int) -> tuple[np.ndarray, memoryview]:
    byte_count = (count * bits + 7) // 8
    if len(blob) < byte_count:
        raise ValueError("truncated unsigned bit stream")
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits).astype(np.uint16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.uint16))[None]
    values = (stream * shifts).sum(axis=1, dtype=np.uint16)
    return values, blob[byte_count:]


def quantized_components(name: str, value: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if value.ndim < 2:
        raise ValueError("quantized_components requires rank >= 2")
    source = value.detach().cpu().float()
    limit = (1 << (bits - 1)) - 1
    embedding = name.endswith("embed.weight")
    reduce_dims = tuple(range(source.ndim - 1)) if embedding else tuple(range(1, source.ndim))
    scales = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
    scales = scales.to(torch.float16).clamp(min=_FP16_MIN_POSITIVE)
    codes = (source / scales.float()).round().clamp(-limit, limit).to(torch.int8)
    restored = codes.float() * scales.float()
    return (
        restored,
        scales.reshape(-1).numpy().astype("<f2", copy=False),
        codes.reshape(-1).numpy().astype(np.int8, copy=False),
    )


def standard_qn_payload(name: str, value: torch.Tensor, bits: int) -> tuple[bytes, torch.Tensor]:
    """Per-axis fp16 scales plus signed ``bits``-wide codes, the SD1M coding."""

    if not 2 <= bits <= 8:
        raise ValueError("semantic bit depth must be in [2, 8]")
    restored, scales, codes = quantized_components(name, value, bits)
    payload = scales.tobytes() + sd1.pack_signed_bits(torch.from_numpy(codes), bits)
    return payload, restored


def standard_q4_payload(name: str, value: torch.Tensor) -> tuple[bytes, torch.Tensor]:
    return standard_qn_payload(name, value, DEFAULT_SEMANTIC_BITS)


def decode_standard_qn(
    name: str, template: torch.Tensor, blob: memoryview, bits: int
) -> tuple[torch.Tensor, memoryview]:
    if not 2 <= bits <= 8:
        raise ValueError("semantic bit depth must be in [2, 8]")
    shape = tuple(template.shape)
    embedding = name.endswith("embed.weight")
    scale_count = shape[-1] if embedding else shape[0]
    scale_bytes = scale_count * 2
    if len(blob) < scale_bytes:
        raise ValueError(f"truncated q{bits} scales for {name}")
    scales = torch.from_numpy(np.frombuffer(blob[:scale_bytes], dtype="<f2").copy()).float()
    remaining = blob[scale_bytes:]
    codes, remaining = sd1.unpack_signed_bits(remaining, template.numel(), bits)
    scale_shape = [1] * template.ndim
    scale_shape[-1 if embedding else 0] = scale_count
    restored = codes.reshape(shape).float() * scales.reshape(scale_shape)
    return restored, remaining


def decode_standard_q4(name: str, template: torch.Tensor, blob: memoryview) -> tuple[torch.Tensor, memoryview]:
    return decode_standard_qn(name, template, blob, DEFAULT_SEMANTIC_BITS)


def fit_codebook(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not 2 <= size <= 256 or flat.size < size:
        raise ValueError("invalid codebook size")
    ordered = np.sort(flat)
    positions = np.rint(np.linspace(0, len(ordered) - 1, size)).astype(np.int64)
    centers = ordered[positions].copy()
    for _ in range(32):
        assignments = np.abs(flat[:, None] - centers[None, :]).argmin(axis=1)
        updated = centers.copy()
        for index in range(size):
            members = flat[assignments == index]
            if members.size:
                updated[index] = members.mean(dtype=np.float64)
        updated.sort()
        if np.array_equal(updated, centers):
            break
        centers = updated
    stored = centers.astype("<f2")
    assignments = np.abs(flat[:, None] - stored.astype(np.float32)[None, :]).argmin(axis=1).astype(np.uint16)
    return stored, assignments


def representation_header(mode: int, arg_a: int, arg_b: int = 0) -> bytes:
    return MAGIC + bytes([VERSION, mode, arg_a, arg_b])


def mask_for_names(names: list[str], selected: set[str]) -> int:
    if not selected.issubset(names):
        raise ValueError("selected tensor is absent from quantized tensor order")
    return sum(1 << index for index, name in enumerate(names) if name in selected)


def canonical_lowrank_factors(value: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = value.shape[0]
    matrix = value.detach().cpu().double().reshape(rows, -1)
    if not 1 <= rank < min(matrix.shape):
        raise ValueError("low-rank candidate does not reduce this matrix")
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :rank]
    singular = singular[:rank]
    vh = vh[:rank]
    for index in range(rank):
        pivot = int(torch.argmax(u[:, index].abs()).item())
        if float(u[pivot, index]) < 0:
            u[:, index] = -u[:, index]
            vh[index] = -vh[index]
    root = singular.sqrt()
    left = (u * root[None]).float()
    right = (root[:, None] * vh).float()
    return left, right


def pack_vq_candidate(
    state: Mapping[str, torch.Tensor], mode: int, codebook_size: int
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    if mode not in {MODE_VECTOR_VQ, MODE_SCALE_VQ, MODE_BOTH_VQ}:
        raise ValueError("invalid VQ mode")
    bits = int(math.log2(codebook_size))
    if (1 << bits) != codebook_size:
        raise ValueError("codebook size must be a power of two")
    vector_names = [name for name, value in state.items() if value.ndim < 2]
    quantized_names = [name for name, value in state.items() if value.ndim >= 2]
    vector_values = np.concatenate([state[name].detach().cpu().float().reshape(-1).numpy() for name in vector_names])
    scale_rows = [quantized_components(name, state[name])[1] for name in quantized_names]
    scale_values = np.concatenate(scale_rows).astype(np.float32)

    payload = bytearray(representation_header(mode, codebook_size))
    vector_codebook = vector_indices = None
    scale_codebook = scale_indices = None
    if mode in {MODE_VECTOR_VQ, MODE_BOTH_VQ}:
        vector_codebook, vector_indices = fit_codebook(vector_values, codebook_size)
        payload.extend(vector_codebook.tobytes())
        payload.extend(pack_unsigned_bits(vector_indices, bits))
    if mode in {MODE_SCALE_VQ, MODE_BOTH_VQ}:
        scale_codebook, scale_indices = fit_codebook(scale_values, codebook_size)
        payload.extend(scale_codebook.tobytes())
        payload.extend(pack_unsigned_bits(scale_indices, bits))

    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    vector_offset = 0
    scale_offset = 0
    for name, value in state.items():
        if value.ndim < 2:
            count = value.numel()
            if vector_codebook is None or vector_indices is None:
                stored = value.detach().cpu().to(torch.float16)
                payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
                expected[name] = stored.float()
            else:
                selected = vector_indices[vector_offset : vector_offset + count]
                restored = vector_codebook[selected].astype(np.float32).reshape(value.shape)
                expected[name] = torch.from_numpy(restored.copy())
                vector_offset += count
            continue

        restored, scales, codes = quantized_components(name, value)
        if scale_codebook is None or scale_indices is None:
            payload.extend(scales.tobytes())
            expected[name] = restored
        else:
            count = len(scales)
            selected = scale_indices[scale_offset : scale_offset + count]
            restored_scales = scale_codebook[selected].astype(np.float32)
            scale_shape = [1] * value.ndim
            scale_shape[-1 if name.endswith("embed.weight") else 0] = count
            expected[name] = torch.from_numpy(codes.copy()).reshape(value.shape).float() * (
                torch.from_numpy(restored_scales.copy()).reshape(scale_shape)
            )
            scale_offset += count
        payload.extend(sd1.pack_signed_bits(torch.from_numpy(codes), 4))

    return (
        bytes(payload),
        expected,
        {
            "mode": mode,
            "codebook_size": codebook_size,
            "vector_values": len(vector_values),
            "scale_values": len(scale_values),
        },
    )


def unpack_vq_candidate(blob: bytes, template: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    version, mode, codebook_size, _ = blob[4:8]
    if version != VERSION or mode not in {MODE_VECTOR_VQ, MODE_SCALE_VQ, MODE_BOTH_VQ}:
        raise ValueError("unsupported SM3R VQ header")
    bits = int(math.log2(codebook_size))
    remaining = memoryview(blob)[8:]
    vector_names = [name for name, value in template.items() if value.ndim < 2]
    quantized_names = [name for name, value in template.items() if value.ndim >= 2]
    vector_count = sum(template[name].numel() for name in vector_names)
    scale_counts = {
        name: (template[name].shape[-1] if name.endswith("embed.weight") else template[name].shape[0])
        for name in quantized_names
    }
    scale_count = sum(scale_counts.values())

    vector_codebook = vector_indices = None
    scale_codebook = scale_indices = None
    if mode in {MODE_VECTOR_VQ, MODE_BOTH_VQ}:
        byte_count = codebook_size * 2
        vector_codebook = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
        remaining = remaining[byte_count:]
        vector_indices, remaining = unpack_unsigned_bits(remaining, vector_count, bits)
    if mode in {MODE_SCALE_VQ, MODE_BOTH_VQ}:
        byte_count = codebook_size * 2
        scale_codebook = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
        remaining = remaining[byte_count:]
        scale_indices, remaining = unpack_unsigned_bits(remaining, scale_count, bits)

    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    vector_offset = 0
    scale_offset = 0
    for name, value in template.items():
        if value.ndim < 2:
            count = value.numel()
            if vector_codebook is None or vector_indices is None:
                byte_count = count * 2
                array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
                remaining = remaining[byte_count:]
            else:
                selected = vector_indices[vector_offset : vector_offset + count]
                array = vector_codebook[selected].copy()
                vector_offset += count
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
            continue

        scale_count_here = scale_counts[name]
        if scale_codebook is None or scale_indices is None:
            byte_count = scale_count_here * 2
            scales = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            remaining = remaining[byte_count:]
        else:
            selected = scale_indices[scale_offset : scale_offset + scale_count_here]
            scales = scale_codebook[selected].copy()
            scale_offset += scale_count_here
        codes, remaining = sd1.unpack_signed_bits(remaining, value.numel(), 4)
        scale_shape = [1] * value.ndim
        scale_shape[-1 if name.endswith("embed.weight") else 0] = scale_count_here
        restored[name] = codes.reshape(value.shape).float() * (torch.from_numpy(scales).float().reshape(scale_shape))
    if remaining:
        raise ValueError(f"SM3R VQ payload has {len(remaining)} trailing bytes")
    return restored


def pack_lowrank_candidate(
    state: Mapping[str, torch.Tensor], rank: int
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    qnames = sd1.quantized_names(state)
    mask = mask_for_names(qnames, LOWRANK_NAMES)
    payload = bytearray(representation_header(MODE_LOWRANK, rank))
    payload.extend(struct.pack("<H", mask))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
            expected[name] = stored.float()
        elif name not in LOWRANK_NAMES:
            encoded, restored = standard_q4_payload(name, value)
            payload.extend(encoded)
            expected[name] = restored
        else:
            left, right = canonical_lowrank_factors(value, rank)
            left_payload, left_restored = standard_q4_payload("factor.left", left)
            right_payload, right_restored = standard_q4_payload("factor.right", right)
            payload.extend(left_payload)
            payload.extend(right_payload)
            expected[name] = (left_restored @ right_restored).reshape(value.shape)
    return (
        bytes(payload),
        expected,
        {
            "rank": rank,
            "selected_tensor_names": sorted(LOWRANK_NAMES),
            "selection_mask": mask,
        },
    )


def unpack_lowrank_candidate(blob: bytes, template: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    version, mode, rank, _ = blob[4:8]
    if version != VERSION or mode != MODE_LOWRANK:
        raise ValueError("unsupported SM3R low-rank header")
    remaining = memoryview(blob)[8:]
    mask = struct.unpack_from("<H", remaining)[0]
    remaining = remaining[2:]
    qnames = sd1.quantized_names(template)
    expected_mask = mask_for_names(qnames, LOWRANK_NAMES)
    if mask != expected_mask:
        raise ValueError("low-rank selection mask differs")
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            byte_count = value.numel() * 2
            array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            remaining = remaining[byte_count:]
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
        elif name not in LOWRANK_NAMES:
            restored[name], remaining = decode_standard_q4(name, value, remaining)
        else:
            rows = value.shape[0]
            columns = value.numel() // rows
            left_template = torch.empty((rows, rank), dtype=torch.float32)
            right_template = torch.empty((rank, columns), dtype=torch.float32)
            left, remaining = decode_standard_q4("factor.left", left_template, remaining)
            right, remaining = decode_standard_q4("factor.right", right_template, remaining)
            restored[name] = (left @ right).reshape(value.shape)
    if remaining:
        raise ValueError(f"SM3R low-rank payload has {len(remaining)} trailing bytes")
    return restored


def pack_prune_candidate(
    state: Mapping[str, torch.Tensor], keep_percent: int
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    if not 1 <= keep_percent < 100:
        raise ValueError("keep percentage must be in [1, 99]")
    qnames = sd1.quantized_names(state)
    mask = mask_for_names(qnames, PRUNE_NAMES)
    payload = bytearray(representation_header(MODE_ROW_PRUNE, keep_percent))
    payload.extend(struct.pack("<H", mask))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    kept_rows: dict[str, int] = {}
    for name, value in state.items():
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
            expected[name] = stored.float()
        elif name not in PRUNE_NAMES:
            encoded, restored = standard_q4_payload(name, value)
            payload.extend(encoded)
            expected[name] = restored
        else:
            rows = value.shape[0]
            keep = max(1, round(rows * keep_percent / 100.0))
            flat = value.detach().cpu().float().reshape(rows, -1)
            norms = flat.square().sum(dim=1)
            order = sorted(range(rows), key=lambda index: (-float(norms[index]), index))
            selected = np.zeros(rows, dtype=np.uint8)
            selected[order[:keep]] = 1
            mask_bytes = np.packbits(selected, bitorder="little").tobytes()
            payload.extend(mask_bytes)
            compact = flat[torch.from_numpy(selected.astype(bool))]
            encoded, restored_compact = standard_q4_payload("pruned.rows", compact)
            payload.extend(encoded)
            restored = torch.zeros_like(flat)
            restored[torch.from_numpy(selected.astype(bool))] = restored_compact
            expected[name] = restored.reshape(value.shape)
            kept_rows[name] = keep
    return (
        bytes(payload),
        expected,
        {
            "keep_percent": keep_percent,
            "selected_tensor_names": sorted(PRUNE_NAMES),
            "kept_rows": kept_rows,
            "selection_mask": mask,
        },
    )


def unpack_prune_candidate(blob: bytes, template: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    version, mode, keep_percent, _ = blob[4:8]
    if version != VERSION or mode != MODE_ROW_PRUNE:
        raise ValueError("unsupported SM3R prune header")
    remaining = memoryview(blob)[8:]
    mask = struct.unpack_from("<H", remaining)[0]
    remaining = remaining[2:]
    qnames = sd1.quantized_names(template)
    if mask != mask_for_names(qnames, PRUNE_NAMES):
        raise ValueError("prune selection mask differs")
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            byte_count = value.numel() * 2
            array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            remaining = remaining[byte_count:]
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
        elif name not in PRUNE_NAMES:
            restored[name], remaining = decode_standard_q4(name, value, remaining)
        else:
            rows = value.shape[0]
            mask_bytes = (rows + 7) // 8
            selected = np.unpackbits(
                np.frombuffer(remaining[:mask_bytes], dtype=np.uint8),
                bitorder="little",
            )[:rows].astype(bool)
            remaining = remaining[mask_bytes:]
            expected_keep = max(1, round(rows * keep_percent / 100.0))
            if int(selected.sum()) != expected_keep:
                raise ValueError(f"prune row count differs for {name}")
            columns = value.numel() // rows
            compact_template = torch.empty((expected_keep, columns), dtype=torch.float32)
            compact, remaining = decode_standard_q4("pruned.rows", compact_template, remaining)
            dense = torch.zeros((rows, columns), dtype=torch.float32)
            dense[torch.from_numpy(selected)] = compact
            restored[name] = dense.reshape(value.shape)
    if remaining:
        raise ValueError(f"SM3R prune payload has {len(remaining)} trailing bytes")
    return restored


def resolve_mixed_depths(
    quantized_order: list[str], depths: Mapping[str, int]
) -> OrderedDict[str, int]:
    """Expand a sparse per-tensor depth override to the full quantized order."""

    unknown = sorted(set(depths) - set(quantized_order))
    if unknown:
        raise ValueError(f"mixed-depth override names are absent from the tensor order: {unknown}")
    resolved: OrderedDict[str, int] = OrderedDict()
    for name in quantized_order:
        bits = int(depths.get(name, DEFAULT_SEMANTIC_BITS))
        if not 2 <= bits <= 8:
            raise ValueError(f"mixed semantic bit depth for {name} must be in [2, 8]")
        resolved[name] = bits
    return resolved


def pack_prune_mixed_candidate(
    state: Mapping[str, torch.Tensor],
    keep_percent: int,
    depths: Mapping[str, int],
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """MODE_ROW_PRUNE plus an SD1M-style per-tensor depth table.

    The row-prune geometry is byte-identical to :func:`pack_prune_candidate`;
    the only structural addition is the depth nibble table, which lets tensors
    outside ``PRUNE_NAMES`` store at fewer than four bits.  Depths also apply to
    the surviving rows of a pruned tensor, so ``depths = {}`` reproduces
    MODE_ROW_PRUNE's payload with an eight-byte allocation-table prefix.
    """

    if not 1 <= keep_percent < 100:
        raise ValueError("keep percentage must be in [1, 99]")
    qnames = sd1.quantized_names(state)
    allocation = resolve_mixed_depths(qnames, depths)
    mask = mask_for_names(qnames, PRUNE_NAMES)
    payload = bytearray(representation_header(MODE_ROW_PRUNE_MIXED, keep_percent))
    payload.extend(struct.pack("<H", mask))
    payload.extend(sd1._pack_depth_nibbles([allocation[name] for name in qnames]))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    kept_rows: dict[str, int] = {}
    for name, value in state.items():
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
            expected[name] = stored.float()
        elif name not in PRUNE_NAMES:
            encoded, restored = standard_qn_payload(name, value, allocation[name])
            payload.extend(encoded)
            expected[name] = restored
        else:
            rows = value.shape[0]
            keep = max(1, round(rows * keep_percent / 100.0))
            flat = value.detach().cpu().float().reshape(rows, -1)
            norms = flat.square().sum(dim=1)
            order = sorted(range(rows), key=lambda index: (-float(norms[index]), index))
            selected = np.zeros(rows, dtype=np.uint8)
            selected[order[:keep]] = 1
            mask_bytes = np.packbits(selected, bitorder="little").tobytes()
            payload.extend(mask_bytes)
            compact = flat[torch.from_numpy(selected.astype(bool))]
            encoded, restored_compact = standard_qn_payload("pruned.rows", compact, allocation[name])
            payload.extend(encoded)
            restored = torch.zeros_like(flat)
            restored[torch.from_numpy(selected.astype(bool))] = restored_compact
            expected[name] = restored.reshape(value.shape)
            kept_rows[name] = keep
    return (
        bytes(payload),
        expected,
        {
            "keep_percent": keep_percent,
            "selected_tensor_names": sorted(PRUNE_NAMES),
            "kept_rows": kept_rows,
            "selection_mask": mask,
            "bit_allocation": dict(allocation),
            "non_default_bit_allocation": {
                name: bits for name, bits in allocation.items() if bits != DEFAULT_SEMANTIC_BITS
            },
        },
    )


def unpack_prune_mixed_candidate(
    blob: bytes, template: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    version, mode, keep_percent, reserved = blob[4:8]
    if version != VERSION or mode != MODE_ROW_PRUNE_MIXED or reserved != 0:
        raise ValueError("unsupported SM3R mixed-prune header")
    if not 1 <= keep_percent < 100:
        raise ValueError("invalid SM3R mixed-prune keep percentage")
    remaining = memoryview(blob)[8:]
    if len(remaining) < 2:
        raise ValueError("truncated mixed-prune selection mask")
    mask = struct.unpack_from("<H", remaining)[0]
    remaining = remaining[2:]
    qnames = sd1.quantized_names(template)
    if mask != mask_for_names(qnames, PRUNE_NAMES):
        raise ValueError("mixed-prune selection mask differs")
    depth_bytes = (len(qnames) + 1) // 2
    if len(remaining) < depth_bytes:
        raise ValueError("truncated mixed-prune depth allocation")
    depths = sd1._unpack_depth_nibbles(bytes(remaining[:depth_bytes]), len(qnames))
    remaining = remaining[depth_bytes:]
    allocation = OrderedDict(zip(qnames, depths, strict=True))
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            byte_count = value.numel() * 2
            array = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            remaining = remaining[byte_count:]
            restored[name] = torch.from_numpy(array.reshape(value.shape)).float()
        elif name not in PRUNE_NAMES:
            restored[name], remaining = decode_standard_qn(name, value, remaining, allocation[name])
        else:
            rows = value.shape[0]
            mask_bytes = (rows + 7) // 8
            selected = np.unpackbits(
                np.frombuffer(remaining[:mask_bytes], dtype=np.uint8),
                bitorder="little",
            )[:rows].astype(bool)
            remaining = remaining[mask_bytes:]
            expected_keep = max(1, round(rows * keep_percent / 100.0))
            if int(selected.sum()) != expected_keep:
                raise ValueError(f"mixed-prune row count differs for {name}")
            columns = value.numel() // rows
            compact_template = torch.empty((expected_keep, columns), dtype=torch.float32)
            compact, remaining = decode_standard_qn(
                "pruned.rows", compact_template, remaining, allocation[name]
            )
            dense = torch.zeros((rows, columns), dtype=torch.float32)
            dense[torch.from_numpy(selected)] = compact
            restored[name] = dense.reshape(value.shape)
    if remaining:
        raise ValueError(f"SM3R mixed-prune payload has {len(remaining)} trailing bytes")
    return restored


def unpack_candidate(blob: bytes, template: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if not blob.startswith(MAGIC) or len(blob) < 8:
        raise ValueError("not an SM3R payload")
    mode = blob[5]
    if mode in {MODE_VECTOR_VQ, MODE_SCALE_VQ, MODE_BOTH_VQ}:
        return unpack_vq_candidate(blob, template)
    if mode == MODE_LOWRANK:
        return unpack_lowrank_candidate(blob, template)
    if mode == MODE_ROW_PRUNE:
        return unpack_prune_candidate(blob, template)
    if mode == MODE_ROW_PRUNE_MIXED:
        return unpack_prune_mixed_candidate(blob, template)
    raise ValueError(f"unknown SM3R mode {mode}")


def state_wire(state: Mapping[str, torch.Tensor]) -> bytes:
    output = bytearray(b"SM3STATE\x01")
    output.extend(struct.pack("<I", len(state)))
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode()
        shape = tuple(tensor.shape)
        data = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        output.extend(struct.pack("<H", len(name_bytes)))
        output.extend(name_bytes)
        output.extend(struct.pack("<B", len(shape)))
        output.extend(struct.pack(f"<{len(shape)}I", *shape))
        output.extend(struct.pack("<Q", tensor.numel() * 4))
        output.extend(data)
    return bytes(output)


def assert_state_equal(expected: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]) -> None:
    if list(expected) != list(actual):
        raise ValueError("decoded state order differs")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            delta = float((expected[name] - actual[name]).abs().max())
            raise ValueError(f"decoded state differs for {name}: max_abs={delta}")


def decompose_baseline(state: Mapping[str, torch.Tensor], out_dir: Path, semantic_blob: bytes) -> dict[str, Any]:
    tensor_dir = out_dir / "retained/baseline/tensors"
    rows = []
    code_bytes = 0
    scale_bytes = 0
    fp16_vector_bytes = 0
    concatenated = bytearray()
    for index, (name, value) in enumerate(state.items()):
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload = stored.numpy().astype("<f2", copy=False).tobytes()
            kind = "fp16_vector"
            tensor_scale_bytes = 0
            tensor_code_bytes = 0
            fp16_vector_bytes += value.numel() * 2
        else:
            restored, scales, codes = quantized_components(name, value)
            del restored
            scales_payload = scales.tobytes()
            codes_payload = sd1.pack_signed_bits(torch.from_numpy(codes), 4)
            payload = scales_payload + codes_payload
            kind = "per_axis_q4"
            tensor_scale_bytes = scales.size * 2
            tensor_code_bytes = (value.numel() * 4 + 7) // 8
            scale_bytes += tensor_scale_bytes
            code_bytes += tensor_code_bytes
        safe = name.replace(".", "_")
        raw_path = tensor_dir / f"{index:02d}_{safe}.bin"
        compressed_path = tensor_dir / f"{index:02d}_{safe}.brotli_q11"
        compressed = brotli.compress(payload, quality=11)
        retain_payload(raw_path, payload)
        retain_payload(compressed_path, compressed)
        concatenated.extend(payload)
        rows.append(
            {
                "index": index,
                "name": name,
                "shape": list(value.shape),
                "parameters": value.numel(),
                "kind": kind,
                "raw_bytes": raw_path.stat().st_size,
                "scale_bytes": tensor_scale_bytes,
                "code_bytes": tensor_code_bytes,
                "standalone_brotli_q11_bytes": compressed_path.stat().st_size,
                "raw_sha256": sha256_bytes(payload),
                "standalone_brotli_q11_sha256": sha256_bytes(compressed),
            }
        )
    if bytes(concatenated) != semantic_blob:
        raise ValueError("per-tensor decomposition does not reconstruct semantic blob")
    semantic_path = out_dir / "retained/baseline/semantic.bin"
    semantic_brotli_path = out_dir / "retained/baseline/semantic.brotli_q11"
    retain_payload(semantic_path, semantic_blob)
    semantic_brotli = brotli.compress(semantic_blob, quality=11)
    retain_payload(semantic_brotli_path, semantic_brotli)
    return {
        "parameter_denominator": sum(value.numel() for value in state.values()),
        "rank_ge_2_parameter_denominator": sum(value.numel() for value in state.values() if value.ndim >= 2),
        "rank_lt_2_parameter_denominator": sum(value.numel() for value in state.values() if value.ndim < 2),
        "raw_semantic_bytes": len(semantic_blob),
        "q4_code_bytes": code_bytes,
        "fp16_scale_bytes": scale_bytes,
        "fp16_vector_bytes": fp16_vector_bytes,
        "naive_all_parameter_q4_bytes": sum(value.numel() for value in state.values()) / 2,
        "standalone_semantic_brotli_q11_bytes": semantic_brotli_path.stat().st_size,
        "exact_archive_leave_one_out_marginal_bytes_prior_receipt": 36_580,
        "exact_archive_leave_one_out_source": (".omx/research/ddm_pr130_reproduce_20260809/PR130_REPRODUCED_HERE.md"),
        "tensors": rows,
        "retained": {
            "semantic": artifact(semantic_path),
            "semantic_brotli_q11": artifact(semantic_brotli_path),
        },
    }


def retain_candidate(
    candidate_id: str,
    semantic_blob: bytes,
    expected_state: Mapping[str, torch.Tensor],
    base: sd1.BaseArchive,
    template: Mapping[str, torch.Tensor],
    out_dir: Path,
    details: Mapping[str, Any],
    *,
    parser: str,
) -> dict[str, Any]:
    candidate_dir = out_dir / "retained/candidates" / candidate_id
    archive_bytes = sd1.rebuild_archive(base, semantic_blob)
    repeat_bytes = sd1.rebuild_archive(base, semantic_blob)
    if archive_bytes != repeat_bytes:
        raise ValueError(f"candidate {candidate_id} is not deterministic")
    extracted = sd1.semantic_blob_from_archive(archive_bytes, base)
    if extracted != semantic_blob:
        raise ValueError(f"candidate {candidate_id} semantic parse-back differs")
    if parser == "sm3r":
        decoded = unpack_candidate(extracted, template)
    elif parser == "sd1":
        decoded, _, _ = sd1.unpack_semantic_state(extracted, template)
    else:
        raise ValueError(f"unknown parser {parser}")
    assert_state_equal(expected_state, decoded)
    decoded_wire = state_wire(decoded)

    semantic_path = candidate_dir / "semantic.bin"
    archive_path = candidate_dir / "archive.zip"
    repeat_path = candidate_dir / "archive.repeat.zip"
    decoded_path = candidate_dir / "decoded_state.sm3state"
    retain_payload(semantic_path, semantic_blob)
    retain_payload(archive_path, archive_bytes)
    retain_payload(repeat_path, repeat_bytes)
    retain_payload(decoded_path, decoded_wire)
    receipt = {
        "schema": "ddm_sm3_retained_candidate.v1",
        "candidate_id": candidate_id,
        "format": parser,
        "score_claim": False,
        "d_seg_measured": False,
        "d_pose_measured": False,
        "public_receiver_readable": parser == "sd1",
        "public_receiver_proof_commit": SD1_PUBLIC_RECEIVER_COMMIT if parser == "sd1" else None,
        "archive_delta_bytes": len(archive_bytes) - EXPECTED_BASE_ARCHIVE_BYTES,
        "rate_only_delta_s": 25 * (len(archive_bytes) - EXPECTED_BASE_ARCHIVE_BYTES) / ORIGINAL_BYTES,
        "break_even_max_d_seg_increase_if_d_pose_unchanged": (
            -25 * (len(archive_bytes) - EXPECTED_BASE_ARCHIVE_BYTES) / ORIGINAL_BYTES / 100
        ),
        "details": dict(details),
        "retained": {
            "semantic": artifact(semantic_path),
            "archive": artifact(archive_path),
            "archive_repeat": artifact(repeat_path),
            "decoded_state": artifact(decoded_path),
        },
        "checks": {
            "complete_archive_rebuilt": True,
            "archive_repeat_byte_equal": True,
            "semantic_parseback_byte_equal": True,
            "all_tensor_decode_equal_to_packer_state": True,
            "tensor_denominator": len(template),
        },
    }
    receipt_path = candidate_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt["receipt"] = artifact(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=250_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = require_ssd(args.out_dir)
    resume_path = require_ssd(args.resume_from)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_dir).free < args.minimum_free_bytes:
        raise RuntimeError("storage preflight refused DDM-SM3 materialization")
    if args.base_archive.stat().st_size != EXPECTED_BASE_ARCHIVE_BYTES:
        raise ValueError("base archive byte count differs")
    if sha256_file(args.base_archive) != EXPECTED_BASE_ARCHIVE_SHA256:
        raise ValueError("base archive SHA-256 differs")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("semantic checkpoint SHA-256 differs")

    torch.manual_seed(20260810)
    np.random.seed(20260810)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state: OrderedDict[str, torch.Tensor] = OrderedDict(checkpoint["state_dict"])
    base = sd1.read_base_archive(args.base_archive)
    measurement_source_sha256 = sha256_file(Path(__file__).resolve())
    reused_sd1_source_sha256 = sha256_file(Path(sd1.__file__).resolve())
    progress = {
        "schema": "ddm_sm3_resume.v1",
        "updated_at_utc": utc_now(),
        "complete": False,
        "score_claim": False,
        "base_archive_sha256": EXPECTED_BASE_ARCHIVE_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "measurement_source_sha256": measurement_source_sha256,
        "reused_sd1_source_sha256": reused_sd1_source_sha256,
        "completed_candidates": [],
    }
    if resume_path.exists():
        prior = json.loads(resume_path.read_text())
        for key in (
            "schema",
            "base_archive_sha256",
            "checkpoint_sha256",
            "measurement_source_sha256",
            "reused_sd1_source_sha256",
        ):
            if prior.get(key) != progress[key]:
                raise ValueError(f"resume binding differs: {key}")
        progress = prior
        progress["complete"] = False
        progress["updated_at_utc"] = utc_now()
    atomic_write_json(resume_path, progress)

    decomposition = decompose_baseline(state, out_dir, base.semantic_blob)
    decomposition_path = out_dir / "SECTION_DECOMPOSITION.json"
    atomic_write_json(decomposition_path, decomposition)

    candidates: list[dict[str, Any]] = []
    qnames = sd1.quantized_names(state)
    baseline_allocation = OrderedDict((name, 4) for name in qnames)
    baseline_blob, baseline_state = sd1.pack_semantic_state(state, baseline_allocation, legacy_int4=True)
    if baseline_blob != base.semantic_blob:
        raise ValueError("legacy q4 positive control differs from shipped semantic bytes")
    if "legacy_q4_control" in progress["completed_candidates"]:
        candidates.append(load_retained_candidate("legacy_q4_control", out_dir))
    else:
        candidates.append(
            retain_candidate(
                "legacy_q4_control",
                baseline_blob,
                baseline_state,
                base,
                state,
                out_dir,
                {"mechanism": "shipped legacy q4 positive control"},
                parser="sd1",
            )
        )
        progress["completed_candidates"].append("legacy_q4_control")
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(resume_path, progress)

    selected_allocation = OrderedDict((name, 3 if name in SELECTED_MIXED_Q3_NAMES else 4) for name in qnames)
    selected_blob, selected_state = sd1.pack_semantic_state(state, selected_allocation, legacy_int4=False)
    if "sd1_selected_mixed_q3q4" in progress["completed_candidates"]:
        candidates.append(load_retained_candidate("sd1_selected_mixed_q3q4", out_dir))
    else:
        candidates.append(
            retain_candidate(
                "sd1_selected_mixed_q3q4",
                selected_blob,
                selected_state,
                base,
                state,
                out_dir,
                {
                    "mechanism": "measured SD1 per-tensor bit reallocation",
                    "q3_tensor_names": sorted(SELECTED_MIXED_Q3_NAMES),
                    "prior_n600_d_seg": 0.00028756883409288193,
                    "prior_axis": "[macOS-CPU advisory; retained official-Ada target; n600]",
                    "prior_source_commit": "600af8ef7d5f4573f6b3793d7a946fe5bf10d4d5",
                },
                parser="sd1",
            )
        )
        progress["completed_candidates"].append("sd1_selected_mixed_q3q4")
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(resume_path, progress)

    builders = [
        ("vector_vq32", lambda: pack_vq_candidate(state, MODE_VECTOR_VQ, 32)),
        ("scale_vq32", lambda: pack_vq_candidate(state, MODE_SCALE_VQ, 32)),
        ("vector_scale_vq32", lambda: pack_vq_candidate(state, MODE_BOTH_VQ, 32)),
        ("pointwise_lowrank_r32", lambda: pack_lowrank_candidate(state, 32)),
        ("film_row_prune_keep75", lambda: pack_prune_candidate(state, 75)),
        ("film_row_prune_keep50", lambda: pack_prune_candidate(state, 50)),
    ]
    for candidate_id, builder in builders:
        if candidate_id in progress["completed_candidates"]:
            candidates.append(load_retained_candidate(candidate_id, out_dir))
        else:
            semantic_blob, expected_state, details = builder()
            candidates.append(
                retain_candidate(
                    candidate_id,
                    semantic_blob,
                    expected_state,
                    base,
                    state,
                    out_dir,
                    details,
                    parser="sm3r",
                )
            )
            progress["completed_candidates"].append(candidate_id)
            progress["updated_at_utc"] = utc_now()
            atomic_write_json(resume_path, progress)

    summary = {
        "schema": "ddm_sm3_semantic_representation_result.v1",
        "written_at_utc": utc_now(),
        "score_claim": False,
        "axis": "[scorer-free exact archive bytes and parse-back]",
        "base": {
            "archive": artifact(args.base_archive),
            "checkpoint": artifact(args.checkpoint),
        },
        "provenance": {
            "seed": 20260810,
            "argv": sys.argv,
            "measurement_source": artifact(Path(__file__).resolve()),
            "reused_sd1_source": artifact(Path(sd1.__file__).resolve()),
            "reused_sd1_commit": "600af8ef7d5f4573f6b3793d7a946fe5bf10d4d5",
            "pr130_intake_commit": "e34f31bc4969042c0051ac81aa3c56884419a231",
        },
        "decomposition": decomposition,
        "decomposition_receipt": artifact(decomposition_path),
        "candidates": candidates,
        "selection_status": "NO_WINNER_WITHOUT_DSEG_AND_DPOSE_MEASUREMENT",
        "scorer_constraint": (
            "DDM-SM3 charter forbids racing ddm_ai1 for the scorer; new SM3R candidates "
            "are retained and queued, not ranked by byte count alone"
        ),
    }
    result_path = out_dir / "SM3_RESULT.json"
    atomic_write_json(result_path, summary)
    progress["complete"] = True
    progress["result"] = artifact(result_path)
    progress["updated_at_utc"] = utc_now()
    atomic_write_json(resume_path, progress)
    print(
        json.dumps(
            {
                "complete": True,
                "result": str(result_path),
                "candidate_count": len(candidates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
