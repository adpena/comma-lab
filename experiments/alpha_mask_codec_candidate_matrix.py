#!/usr/bin/env python3
"""Build a deterministic Alpha mask codec candidate matrix.

This tool is planning evidence only. It reads one archive mask member with
strict ZIP custody checks, decodes the mask tensor through local codec helpers,
then emits charged representation artifacts that a later archive builder may
integrate and exact CUDA auth eval must validate.

No scorer network is loaded and no score claim is made.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCHEMA = "alpha_mask_codec_candidate_matrix_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this matrix. Candidate artifacts are byte and "
    "round-trip planning evidence only; a deterministic finalist archive and "
    "exact CUDA auth eval are required before promotion, ranking, retirement, "
    "or score claims."
)

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/alpha_mask_codec_candidate_matrix"
DEFAULT_MAX_FRAMES = 64
CLASS_IDS = (0, 1, 2, 3, 4)
FOREGROUND_CLASS_IDS = (1, 2, 3, 4)
IMPLICIT_CLASS_ID = 0

MANIFEST_NAME = "alpha_mask_codec_candidate_matrix.json"
COCO_RLE_MEMBER = "coco_rle_runs.amcrle"
COMPONENT_BOUNDARY_MEMBER = "component_boundary_delta.amccb"
TRANSITION_ENDPOINT_MEMBER = "class_transition_endpoints.amcte"
PALETTE_PNG_MEMBER = "palette_png_sequence.zip"
AV1_MONO_MEMBER = "av1_monochrome_reference.mkv"

_HIDDEN_SYSTEM_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db"}
_HEADER_LEN_STRUCT = ">I"
_CONTAINER_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class MatrixConfig:
    max_frames: int | None = DEFAULT_MAX_FRAMES
    families: tuple[str, ...] = (
        "coco_rle",
        "component_boundary_delta",
        "transition_endpoints",
        "palette_png_sequence",
        "av1_monochrome_reference",
    )
    compression: str = "zlib"
    zlib_level: int = 9
    av1_crf: int = 50
    av1_fps: int = 20
    nrv_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ComponentRun:
    y: int
    x0: int
    x1: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    member_path = PurePosixPath(name)
    parts = member_path.parts
    if member_path.is_absolute() or ".." in parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if not parts or any(part in ("", ".") for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    first = parts[0]
    if len(first) == 2 and first[1] == ":":
        raise ValueError(f"unsafe archive member path: {name!r}")
    return parts


def _reject_hidden_or_system_member(name: str, parts: tuple[str, ...]) -> None:
    if any(part in _HIDDEN_SYSTEM_NAMES for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")


def _validate_requested_member(member: str) -> None:
    parts = _safe_member_parts(member)
    _reject_hidden_or_system_member(member, parts)


def _validated_zip_infos(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        if info.filename in infos:
            raise ValueError(f"duplicate archive member: {info.filename!r}")
        if info.is_dir():
            raise ValueError(f"unsafe archive directory member: {info.filename!r}")
        parts = _safe_member_parts(info.filename)
        _reject_hidden_or_system_member(info.filename, parts)
        infos[info.filename] = info
    return infos


def _member_inventory(infos: dict[str, zipfile.ZipInfo]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "size_bytes": int(info.file_size),
            "compressed_size_bytes": int(info.compress_size),
            "crc32": f"{info.CRC:08x}",
        }
        for name, info in sorted(infos.items())
    ]


def _read_archive_member(archive: Path, member: str) -> tuple[bytes, dict[str, Any]]:
    archive = Path(archive)
    _validate_requested_member(member)
    with zipfile.ZipFile(archive, "r") as zf:
        infos = _validated_zip_infos(zf)
        if member not in infos:
            raise FileNotFoundError(f"{archive} missing archive member {member!r}")
        info = infos[member]
        data = zf.read(info)

    return data, {
        "archive_path": str(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "archive_sha256": _sha256_file(archive),
        "member_inventory": _member_inventory(infos),
        "mask_member": {
            "name": member,
            "size_bytes": int(info.file_size),
            "compressed_size_bytes": int(info.compress_size),
            "crc32": f"{info.CRC:08x}",
            "sha256": _sha256_bytes(data),
        },
    }


def _validate_masks(masks: torch.Tensor) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"masks must be a torch.Tensor, got {type(masks).__name__}")
    if masks.dim() != 3:
        raise ValueError(f"masks must have shape (T,H,W), got {tuple(masks.shape)}")
    if masks.numel() == 0:
        raise ValueError("masks tensor is empty")
    masks = masks.detach().cpu().to(torch.int64).contiguous()
    min_value = int(masks.min().item())
    max_value = int(masks.max().item())
    if min_value < min(CLASS_IDS) or max_value > max(CLASS_IDS):
        raise ValueError(f"masks must contain class ids in [0,4], got [{min_value},{max_value}]")
    if masks.shape[0] > 0xFFFFFFFF:
        raise ValueError(f"frame count too large for candidate packets: {masks.shape[0]}")
    if masks.shape[1] > 0xFFFFFFFF or masks.shape[2] > 0xFFFFFFFF:
        raise ValueError(f"mask dimensions too large for candidate packets: {tuple(masks.shape)}")
    return masks


def _tensor_u8_sha256(masks: torch.Tensor) -> str:
    masks = _validate_masks(masks)
    return _sha256_bytes(masks.to(torch.uint8).contiguous().numpy().tobytes())


def _mask_stats(masks: torch.Tensor) -> dict[str, Any]:
    masks = _validate_masks(masks)
    counts = torch.bincount(masks.reshape(-1), minlength=max(CLASS_IDS) + 1).to(torch.int64)
    total = int(masks.numel())
    vertical_edges = int((masks[:, 1:, :] != masks[:, :-1, :]).sum().item()) if masks.shape[1] > 1 else 0
    horizontal_edges = int((masks[:, :, 1:] != masks[:, :, :-1]).sum().item()) if masks.shape[2] > 1 else 0
    temporal_changes = int((masks[1:] != masks[:-1]).sum().item()) if masks.shape[0] > 1 else 0
    return {
        "shape": [int(v) for v in masks.shape],
        "dtype": str(masks.dtype),
        "num_pixels": total,
        "class_histogram": {str(class_id): int(counts[class_id].item()) for class_id in CLASS_IDS},
        "class_fractions": {
            str(class_id): _round_float(int(counts[class_id].item()) / total) for class_id in CLASS_IDS
        },
        "spatial_boundary_edges_4conn": vertical_edges + horizontal_edges,
        "temporal_changed_pixels": temporal_changes,
        "class_id_u8_sha256": _tensor_u8_sha256(masks),
    }


def _agreement_metrics(source: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    source = _validate_masks(source)
    candidate = _validate_masks(candidate)
    if tuple(source.shape) != tuple(candidate.shape):
        return {
            "shape_match": False,
            "source_shape": [int(v) for v in source.shape],
            "candidate_shape": [int(v) for v in candidate.shape],
            "argmax_agreement": None,
            "argmax_disagreement": None,
            "different_pixels": None,
            "num_pixels": int(source.numel()),
        }
    diff = source != candidate
    different = int(diff.sum().item())
    total = int(source.numel())
    return {
        "shape_match": True,
        "source_shape": [int(v) for v in source.shape],
        "candidate_shape": [int(v) for v in candidate.shape],
        "num_pixels": total,
        "equal_pixels": total - different,
        "different_pixels": different,
        "argmax_agreement": _round_float(1.0 - (different / total)),
        "argmax_disagreement": _round_float(different / total),
        "exact_reconstruction": different == 0,
        "candidate_class_id_u8_sha256": _tensor_u8_sha256(candidate),
    }


def _encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"uvarint cannot encode negative value {value}")
    out = bytearray()
    n = int(value)
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated uvarint stream")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("uvarint is too large")


def _pack_varints(values: Iterable[int]) -> bytes:
    payload = bytearray()
    for value in values:
        payload.extend(_encode_uvarint(int(value)))
    return bytes(payload)


def _compress_body(body: bytes, *, config: MatrixConfig) -> tuple[bytes, dict[str, Any]]:
    if config.compression == "none":
        return body, {"codec": "none", "level": None}
    if config.compression == "zlib":
        compressed = zlib.compress(body, level=int(config.zlib_level))
        return compressed, {"codec": "zlib", "level": int(config.zlib_level)}
    raise ValueError(f"unsupported compression {config.compression!r}")


def _decompress_body(body: bytes, *, codec: str) -> bytes:
    if codec == "none":
        return body
    if codec == "zlib":
        return zlib.decompress(body)
    raise ValueError(f"unsupported body codec {codec!r}")


def _packet_bytes(
    *,
    magic: bytes,
    header: dict[str, Any],
    body: bytes,
    config: MatrixConfig,
) -> bytes:
    compressed_body, compression_meta = _compress_body(body, config=config)
    full_header = {
        **header,
        "magic": magic.decode("ascii"),
        "header_struct": _HEADER_LEN_STRUCT,
        "body_compression": compression_meta,
        "body_uncompressed_size_bytes": len(body),
        "body_uncompressed_sha256": _sha256_bytes(body),
        "body_size_bytes": len(compressed_body),
        "body_sha256": _sha256_bytes(compressed_body),
    }
    header_bytes = json.dumps(full_header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return magic + struct.pack(_HEADER_LEN_STRUCT, len(header_bytes)) + header_bytes + compressed_body


def _unpack_packet(payload: bytes, *, expected_magic: bytes) -> tuple[dict[str, Any], bytes]:
    if not payload.startswith(expected_magic):
        raise ValueError(f"payload missing {expected_magic!r} magic")
    offset = len(expected_magic)
    if len(payload) < offset + struct.calcsize(_HEADER_LEN_STRUCT):
        raise ValueError("payload missing header length")
    (header_len,) = struct.unpack(_HEADER_LEN_STRUCT, payload[offset : offset + 4])
    offset += 4
    header_end = offset + int(header_len)
    if header_end > len(payload):
        raise ValueError("payload header extends past payload")
    header = json.loads(payload[offset:header_end].decode("utf-8"))
    body_payload = payload[header_end:]
    compression = header.get("body_compression", {})
    body = _decompress_body(body_payload, codec=str(compression.get("codec", "none")))
    if _sha256_bytes(body) != header.get("body_uncompressed_sha256"):
        raise ValueError("uncompressed body sha256 mismatch")
    return header, body


def _coco_counts_for_binary(binary: np.ndarray) -> list[int]:
    flat = binary.astype(np.uint8, copy=False).reshape(-1)
    if flat.size == 0:
        return [0]
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [flat.size]))
    lengths = (ends - starts).astype(np.int64).tolist()
    if int(flat[0]) == 1:
        return [0, *lengths]
    return lengths


def _encode_coco_rle(masks: torch.Tensor, *, config: MatrixConfig) -> bytes:
    masks_np = _validate_masks(masks).to(torch.uint8).numpy()
    t, h, w = [int(v) for v in masks_np.shape]
    body = bytearray()
    total_runs = 0
    for frame_idx in range(t):
        frame = masks_np[frame_idx]
        for class_id in FOREGROUND_CLASS_IDS:
            counts = _coco_counts_for_binary(frame == class_id)
            total_runs += len(counts)
            body.extend(_encode_uvarint(len(counts)))
            body.extend(_pack_varints(counts))
    return _packet_bytes(
        magic=b"AMRL",
        header={
            "schema": "alpha_mask_coco_rle_runs_v1",
            "shape": [t, h, w],
            "class_ids": list(CLASS_IDS),
            "encoded_class_ids": list(FOREGROUND_CLASS_IDS),
            "implicit_class_id": IMPLICIT_CLASS_ID,
            "scan_order": "frame,class,row-major",
            "rle_convention": "COCO-style counts starting with zero run",
            "total_rle_count_entries": int(total_runs),
        },
        body=bytes(body),
        config=config,
    )


def _decode_coco_rle(payload: bytes) -> torch.Tensor:
    header, body = _unpack_packet(payload, expected_magic=b"AMRL")
    t, h, w = [int(v) for v in header["shape"]]
    out = np.zeros((t, h, w), dtype=np.uint8)
    offset = 0
    frame_size = h * w
    for frame_idx in range(t):
        flat = out[frame_idx].reshape(-1)
        for class_id in header["encoded_class_ids"]:
            count_len, offset = _decode_uvarint(body, offset)
            pos = 0
            foreground = False
            for _ in range(count_len):
                run_len, offset = _decode_uvarint(body, offset)
                end = pos + run_len
                if end > frame_size:
                    raise ValueError("COCO RLE run extends past frame")
                if foreground and run_len:
                    flat[pos:end] = int(class_id)
                pos = end
                foreground = not foreground
            if pos != frame_size:
                raise ValueError("COCO RLE did not cover full frame")
    if offset != len(body):
        raise ValueError("COCO RLE payload has trailing bytes")
    return torch.from_numpy(out.astype(np.int64, copy=False))


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def make(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        return idx

    def find(self, idx: int) -> int:
        parent = self.parent[idx]
        if parent != idx:
            self.parent[idx] = self.find(parent)
        return self.parent[idx]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _row_runs(binary_frame: np.ndarray) -> list[ComponentRun]:
    runs: list[ComponentRun] = []
    for y in range(binary_frame.shape[0]):
        xs = np.flatnonzero(binary_frame[y])
        if xs.size == 0:
            continue
        split_points = np.flatnonzero(np.diff(xs) != 1) + 1
        for segment in np.split(xs, split_points):
            if segment.size:
                runs.append(ComponentRun(y=int(y), x0=int(segment[0]), x1=int(segment[-1]) + 1))
    return runs


def _connected_components_as_runs(binary_frame: np.ndarray) -> list[list[ComponentRun]]:
    runs = _row_runs(binary_frame)
    if not runs:
        return []
    dsu = _DisjointSet()
    run_ids = [dsu.make() for _ in runs]
    prev_row: list[tuple[int, ComponentRun]] = []
    current_row_index = -1
    current_row: list[tuple[int, ComponentRun]] = []
    for idx, run in enumerate(runs):
        if run.y != current_row_index:
            prev_row = current_row if current_row_index == run.y - 1 else []
            current_row = []
            current_row_index = run.y
        for prev_idx, prev in prev_row:
            if prev.x0 < run.x1 and run.x0 < prev.x1:
                dsu.union(run_ids[idx], run_ids[prev_idx])
        current_row.append((idx, run))

    grouped: dict[int, list[ComponentRun]] = {}
    for idx, run in enumerate(runs):
        grouped.setdefault(dsu.find(run_ids[idx]), []).append(run)
    components = [sorted(items, key=lambda r: (r.y, r.x0, r.x1)) for items in grouped.values()]
    return sorted(components, key=lambda comp: (comp[0].y, min(r.x0 for r in comp), len(comp)))


def _encode_component_boundary_delta(masks: torch.Tensor, *, config: MatrixConfig) -> bytes:
    masks_np = _validate_masks(masks).to(torch.uint8).numpy()
    t, h, w = [int(v) for v in masks_np.shape]
    body = bytearray()
    total_components = 0
    total_spans = 0
    for frame_idx in range(t):
        frame = masks_np[frame_idx]
        for class_id in FOREGROUND_CLASS_IDS:
            components = _connected_components_as_runs(frame == class_id)
            total_components += len(components)
            body.extend(_encode_uvarint(len(components)))
            for component in components:
                y0 = min(run.y for run in component)
                y1 = max(run.y for run in component) + 1
                x0 = min(run.x0 for run in component)
                x1 = max(run.x1 for run in component)
                body.extend(
                    _pack_varints(
                        [
                            y0,
                            x0,
                            y1 - y0,
                            x1 - x0,
                            len(component),
                        ]
                    )
                )
                total_spans += len(component)
                for run in component:
                    body.extend(_pack_varints([run.y - y0, run.x0 - x0, run.x1 - run.x0]))
    return _packet_bytes(
        magic=b"AMCB",
        header={
            "schema": "alpha_mask_connected_component_boundary_delta_v1",
            "shape": [t, h, w],
            "class_ids": list(CLASS_IDS),
            "encoded_class_ids": list(FOREGROUND_CLASS_IDS),
            "implicit_class_id": IMPLICIT_CLASS_ID,
            "connectivity": "4-connected per class per frame",
            "component_payload": "bbox plus row-span offsets",
            "total_components": int(total_components),
            "total_row_spans": int(total_spans),
        },
        body=bytes(body),
        config=config,
    )


def _decode_component_boundary_delta(payload: bytes) -> torch.Tensor:
    header, body = _unpack_packet(payload, expected_magic=b"AMCB")
    t, h, w = [int(v) for v in header["shape"]]
    out = np.zeros((t, h, w), dtype=np.uint8)
    offset = 0
    for frame_idx in range(t):
        for class_id in header["encoded_class_ids"]:
            component_count, offset = _decode_uvarint(body, offset)
            for _ in range(component_count):
                y0, offset = _decode_uvarint(body, offset)
                x0, offset = _decode_uvarint(body, offset)
                height, offset = _decode_uvarint(body, offset)
                width, offset = _decode_uvarint(body, offset)
                span_count, offset = _decode_uvarint(body, offset)
                if y0 + height > h or x0 + width > w:
                    raise ValueError("component bbox extends outside frame")
                for _span in range(span_count):
                    dy, offset = _decode_uvarint(body, offset)
                    dx0, offset = _decode_uvarint(body, offset)
                    length, offset = _decode_uvarint(body, offset)
                    y = y0 + dy
                    xs = x0 + dx0
                    xe = xs + length
                    if y >= h or xs < x0 or xe > x0 + width or xe > w or length <= 0:
                        raise ValueError("component span extends outside frame")
                    out[frame_idx, y, xs:xe] = int(class_id)
    if offset != len(body):
        raise ValueError("component payload has trailing bytes")
    return torch.from_numpy(out.astype(np.int64, copy=False))


def _encode_transition_endpoints(masks: torch.Tensor, *, config: MatrixConfig) -> bytes:
    masks_np = _validate_masks(masks).to(torch.uint8).numpy()
    t, h, w = [int(v) for v in masks_np.shape]
    body = bytearray()
    total_transitions = 0
    for frame_idx in range(t):
        for y in range(h):
            row = masks_np[frame_idx, y]
            body.extend(_encode_uvarint(int(row[0])))
            transitions = np.flatnonzero(row[1:] != row[:-1]) + 1
            total_transitions += int(transitions.size)
            body.extend(_encode_uvarint(int(transitions.size)))
            prev_x = 0
            for x in transitions.tolist():
                body.extend(_encode_uvarint(int(x) - prev_x))
                body.extend(_encode_uvarint(int(row[x])))
                prev_x = int(x)
    return _packet_bytes(
        magic=b"AMTE",
        header={
            "schema": "alpha_mask_class_transition_endpoints_v1",
            "shape": [t, h, w],
            "class_ids": list(CLASS_IDS),
            "scan_order": "frame,row,left-to-right",
            "endpoint_semantics": "transition x coordinate where the next class begins",
            "total_transition_endpoints": int(total_transitions),
        },
        body=bytes(body),
        config=config,
    )


def _decode_transition_endpoints(payload: bytes) -> torch.Tensor:
    header, body = _unpack_packet(payload, expected_magic=b"AMTE")
    t, h, w = [int(v) for v in header["shape"]]
    out = np.zeros((t, h, w), dtype=np.uint8)
    offset = 0
    for frame_idx in range(t):
        for y in range(h):
            current_class, offset = _decode_uvarint(body, offset)
            transition_count, offset = _decode_uvarint(body, offset)
            pos = 0
            prev_x = 0
            for _ in range(transition_count):
                dx, offset = _decode_uvarint(body, offset)
                next_class, offset = _decode_uvarint(body, offset)
                x = prev_x + dx
                if x <= pos or x > w:
                    raise ValueError("transition endpoint out of row order")
                out[frame_idx, y, pos:x] = int(current_class)
                current_class = int(next_class)
                pos = x
                prev_x = x
            out[frame_idx, y, pos:w] = int(current_class)
    if offset != len(body):
        raise ValueError("transition endpoint payload has trailing bytes")
    return torch.from_numpy(out.astype(np.int64, copy=False))


def _deterministic_zip_writestr(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_CONTAINER_DATE)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data)


def _encode_palette_png_sequence(masks: torch.Tensor, output_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    pil_spec = importlib.util.find_spec("PIL")
    if pil_spec is None:
        raise RuntimeError("Pillow is not installed; palette PNG sequence skipped")
    from PIL import Image

    masks_np = _validate_masks(masks).to(torch.uint8).numpy()
    palette = [
        0,
        0,
        0,
        255,
        255,
        255,
        64,
        64,
        64,
        192,
        192,
        192,
        128,
        128,
        128,
    ] + [0] * (256 * 3 - 15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as zf:
        for frame_idx, frame in enumerate(masks_np):
            image = Image.fromarray(frame, mode="P")
            image.putpalette(palette)
            with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as tmp:
                image.save(tmp, format="PNG", optimize=False, compress_level=9)
                tmp.seek(0)
                _deterministic_zip_writestr(zf, f"frames/{frame_idx:06d}.png", tmp.read())

    decoded_frames: list[np.ndarray] = []
    with zipfile.ZipFile(output_path, "r") as zf:
        infos = _validated_zip_infos(zf)
        expected_names = [f"frames/{idx:06d}.png" for idx in range(masks_np.shape[0])]
        if sorted(infos) != expected_names:
            raise ValueError("palette PNG ZIP did not contain the expected deterministic frame names")
        for name in expected_names:
            with zf.open(name) as handle:
                image = Image.open(handle)
                decoded_frames.append(np.array(image, dtype=np.uint8))
    decoded = torch.from_numpy(np.stack(decoded_frames, axis=0).astype(np.int64, copy=False))
    return decoded, {
        "encoder": "Pillow palette PNG sequence in deterministic ZIP",
        "zip_compression": "stored",
        "frame_count": int(masks_np.shape[0]),
        "palette_rgb_flat_first_5_classes": palette[:15],
    }


def _encode_av1_monochrome_reference(
    masks: torch.Tensor,
    output_path: Path,
    *,
    config: MatrixConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from tac.mask_codec import decode_masks, encode_masks_monochrome

    masks = _validate_masks(masks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = encode_masks_monochrome(
        masks,
        output_path,
        crf=int(config.av1_crf),
        fps=int(config.av1_fps),
    )
    decoded = decode_masks(output_path, expected_frames=int(masks.shape[0]))
    return _validate_masks(decoded), {
        "encoder": "tac.mask_codec.encode_masks_monochrome",
        "decoder": "tac.mask_codec.decode_masks",
        "crf": int(config.av1_crf),
        "fps": int(config.av1_fps),
        "reported_size_bytes": int(size),
        "diagnostic_only": True,
    }


def _artifact_record(path: Path, *, role: str, archive_member: str | None = None) -> dict[str, Any]:
    record = {
        "role": role,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    if archive_member is not None:
        record["candidate_archive_member"] = archive_member
    return record


def _candidate_record(
    *,
    name: str,
    family: str,
    path: Path,
    archive_member: str,
    source_masks: torch.Tensor,
    decoded_masks: torch.Tensor,
    source_member_size: int | None,
    payload_format: str,
    exact: bool,
    diagnostic: bool,
    details: dict[str, Any],
) -> dict[str, Any]:
    artifact = _artifact_record(path, role=family, archive_member=archive_member)
    size = int(artifact["size_bytes"])
    agreement = _agreement_metrics(source_masks, decoded_masks)
    if exact and agreement["exact_reconstruction"] is not True:
        raise AssertionError(f"{name} was expected to reconstruct exactly")
    record = {
        "name": name,
        "family": family,
        "payload_format": payload_format,
        "charged_representation": True,
        "diagnostic_reference": bool(diagnostic),
        "exact_reconstruction": bool(agreement["exact_reconstruction"]),
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "artifact": artifact,
        "agreement": agreement,
        "runtime_decoder_integration_required": True,
        "details": details,
    }
    if source_member_size is not None:
        record["bytes_vs_source_mask_member"] = {
            "source_mask_member_size_bytes": int(source_member_size),
            "candidate_payload_size_bytes": size,
            "delta_bytes": size - int(source_member_size),
            "under_source_mask_member": bool(size <= int(source_member_size)),
        }
    return record


def _skipped_candidate(name: str, family: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "skipped": True,
        "skip_reason": reason,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
    }


def _selected_environment() -> dict[str, str]:
    keys = [
        "TAC_FFMPEG",
        "TAC_FFPROBE",
        "TAC_UPSTREAM_DIR",
        "PYTHONHASHSEED",
        "UV_PROJECT_ENVIRONMENT",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _module_availability() -> dict[str, bool]:
    modules = {
        "PIL": "pillow_palette_png",
        "tac.mask_codec": "tac_mask_codec",
        "tac.mask_grayscale_lut": "tac_mask_grayscale_lut",
        "tac.nerv_mask_codec": "tac_nerv_mask_codec",
        "numpy": "numpy",
        "torch": "torch",
    }
    return {label: importlib.util.find_spec(module) is not None for module, label in modules.items()}


def _provenance(command: list[str] | None) -> dict[str, Any]:
    return {
        "tool": "experiments/alpha_mask_codec_candidate_matrix.py",
        "command": list(command) if command is not None else list(sys.argv),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": _selected_environment(),
        "module_available": _module_availability(),
        "path_ffmpeg": shutil.which("ffmpeg"),
        "path_ffprobe": shutil.which("ffprobe"),
    }


def _validate_config(config: MatrixConfig) -> None:
    allowed = {
        "coco_rle",
        "component_boundary_delta",
        "transition_endpoints",
        "palette_png_sequence",
        "av1_monochrome_reference",
    }
    if not config.families:
        raise ValueError("at least one candidate family is required")
    unknown = sorted(set(config.families) - allowed)
    if unknown:
        raise ValueError(f"unknown candidate families: {unknown}")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {config.max_frames}")
    if config.compression not in {"none", "zlib"}:
        raise ValueError(f"compression must be 'none' or 'zlib', got {config.compression!r}")
    if not (0 <= config.zlib_level <= 9):
        raise ValueError(f"zlib_level must be in [0,9], got {config.zlib_level}")
    if not (0 <= config.av1_crf <= 63):
        raise ValueError(f"av1_crf must be in [0,63], got {config.av1_crf}")
    if config.av1_fps <= 0:
        raise ValueError(f"av1_fps must be positive, got {config.av1_fps}")
    if config.nrv_shape is not None and any(int(v) <= 0 for v in config.nrv_shape):
        raise ValueError(f"nrv_shape values must be positive, got {config.nrv_shape}")


def _config_record(config: MatrixConfig) -> dict[str, Any]:
    record = dataclasses.asdict(config)
    record["families"] = list(config.families)
    record["nrv_shape"] = list(config.nrv_shape) if config.nrv_shape is not None else None
    return record


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        MANIFEST_NAME,
        COCO_RLE_MEMBER,
        COMPONENT_BOUNDARY_MEMBER,
        TRANSITION_ENDPOINT_MEMBER,
        PALETTE_PNG_MEMBER,
        AV1_MONO_MEMBER,
    ]
    existing = [name for name in targets if (output_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"{output_dir} already contains matrix artifacts {existing}; use --force")


def _rankings(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    completed = [candidate for candidate in candidates if not candidate.get("skipped")]

    def compact(record: dict[str, Any]) -> dict[str, Any]:
        artifact = record["artifact"]
        agreement = record["agreement"]
        return {
            "name": record["name"],
            "family": record["family"],
            "size_bytes": int(artifact["size_bytes"]),
            "sha256": artifact["sha256"],
            "exact_reconstruction": bool(record["exact_reconstruction"]),
            "diagnostic_reference": bool(record["diagnostic_reference"]),
            "argmax_agreement": agreement["argmax_agreement"],
            "different_pixels": agreement["different_pixels"],
            "path": artifact["path"],
            "candidate_archive_member": artifact.get("candidate_archive_member"),
        }

    exact = [
        record for record in completed if record.get("exact_reconstruction") is True and not record.get("diagnostic_reference")
    ]
    diagnostics = [record for record in completed if record.get("diagnostic_reference") is True]
    all_ranked = sorted(
        completed,
        key=lambda record: (
            int(record["artifact"]["size_bytes"]),
            -float(record["agreement"]["argmax_agreement"] or 0.0),
            record["name"],
        ),
    )
    return {
        "exact_reconstruction_by_bytes": [compact(record) for record in sorted(exact, key=lambda r: (r["artifact"]["size_bytes"], r["name"]))],
        "diagnostic_references_by_bytes": [
            compact(record)
            for record in sorted(
                diagnostics,
                key=lambda r: (r["artifact"]["size_bytes"], -float(r["agreement"]["argmax_agreement"] or 0.0), r["name"]),
            )
        ],
        "all_candidates_by_bytes_then_agreement": [compact(record) for record in all_ranked],
    }


def _assert_non_promotable(report: dict[str, Any]) -> None:
    if report.get("score_claim") is not False:
        raise AssertionError("top-level score_claim must be false")
    if report.get("promotion_eligible") is not False:
        raise AssertionError("top-level promotion_eligible must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError(f"evidence_grade must be {EVIDENCE_GRADE!r}")
    if report.get("scorer_network_loaded") is not False:
        raise AssertionError("matrix must not load scorer networks")
    if "contest_auth_eval.py --device cuda" not in report.get("canonical_score_source_required", ""):
        raise AssertionError("report must state exact CUDA auth eval score source")
    for candidate in report.get("candidates", []):
        if candidate.get("score_claim") is not False:
            raise AssertionError(f"candidate score_claim must be false: {candidate.get('name')}")
        if candidate.get("promotion_eligible") is not False:
            raise AssertionError(f"candidate promotion_eligible must be false: {candidate.get('name')}")
        if not candidate.get("skipped"):
            artifact = candidate.get("artifact", {})
            if "sha256" not in artifact or "size_bytes" not in artifact:
                raise AssertionError(f"candidate artifact missing custody fields: {candidate.get('name')}")


def _build_candidate_matrix_from_masks(
    *,
    masks: torch.Tensor,
    source_meta: dict[str, Any],
    output_dir: Path,
    config: MatrixConfig,
    command: list[str] | None = None,
    decode_meta: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    input_masks = _validate_masks(masks)
    effective_decode_meta = dict(decode_meta or {})
    source_masks = input_masks
    if config.max_frames is not None and int(input_masks.shape[0]) > config.max_frames:
        source_masks = input_masks[: int(config.max_frames)].contiguous()
        effective_decode_meta.update(
            {
                "reported_frames": effective_decode_meta.get("reported_frames", int(input_masks.shape[0])),
                "decoded_frames": int(source_masks.shape[0]),
                "max_frames": int(config.max_frames),
                "truncated_by_matrix": True,
            }
        )
    else:
        effective_decode_meta.setdefault("decoded_frames", int(source_masks.shape[0]))
        effective_decode_meta.setdefault("max_frames", config.max_frames)
        effective_decode_meta.setdefault("truncated_by_matrix", False)

    _prepare_output_dir(output_dir, force=force)
    source_member_size = None
    if isinstance(source_meta.get("mask_member"), dict) and source_meta["mask_member"].get("size_bytes") is not None:
        source_member_size = int(source_meta["mask_member"]["size_bytes"])

    candidates: list[dict[str, Any]] = []
    source_sha = _tensor_u8_sha256(source_masks)

    if "coco_rle" in config.families:
        path = output_dir / COCO_RLE_MEMBER
        payload = _encode_coco_rle(source_masks, config=config)
        path.write_bytes(payload)
        decoded = _decode_coco_rle(payload)
        candidates.append(
            _candidate_record(
                name="coco_rle_per_frame_foreground_runs",
                family="coco_rle",
                path=path,
                archive_member=COCO_RLE_MEMBER,
                source_masks=source_masks,
                decoded_masks=decoded,
                source_member_size=source_member_size,
                payload_format="alpha_mask_coco_rle_runs_v1",
                exact=True,
                diagnostic=False,
                details={
                    "encoded_class_ids": list(FOREGROUND_CLASS_IDS),
                    "implicit_class_id": IMPLICIT_CLASS_ID,
                    "source_mask_u8_sha256": source_sha,
                },
            )
        )

    if "component_boundary_delta" in config.families:
        path = output_dir / COMPONENT_BOUNDARY_MEMBER
        payload = _encode_component_boundary_delta(source_masks, config=config)
        path.write_bytes(payload)
        decoded = _decode_component_boundary_delta(payload)
        candidates.append(
            _candidate_record(
                name="connected_component_boundary_delta_packets",
                family="component_boundary_delta",
                path=path,
                archive_member=COMPONENT_BOUNDARY_MEMBER,
                source_masks=source_masks,
                decoded_masks=decoded,
                source_member_size=source_member_size,
                payload_format="alpha_mask_connected_component_boundary_delta_v1",
                exact=True,
                diagnostic=False,
                details={
                    "connectivity": "4-connected",
                    "encoded_class_ids": list(FOREGROUND_CLASS_IDS),
                    "implicit_class_id": IMPLICIT_CLASS_ID,
                    "source_mask_u8_sha256": source_sha,
                },
            )
        )

    if "transition_endpoints" in config.families:
        path = output_dir / TRANSITION_ENDPOINT_MEMBER
        payload = _encode_transition_endpoints(source_masks, config=config)
        path.write_bytes(payload)
        decoded = _decode_transition_endpoints(payload)
        candidates.append(
            _candidate_record(
                name="class_transition_endpoint_packets",
                family="transition_endpoints",
                path=path,
                archive_member=TRANSITION_ENDPOINT_MEMBER,
                source_masks=source_masks,
                decoded_masks=decoded,
                source_member_size=source_member_size,
                payload_format="alpha_mask_class_transition_endpoints_v1",
                exact=True,
                diagnostic=False,
                details={
                    "scan_order": "frame,row,left-to-right",
                    "source_mask_u8_sha256": source_sha,
                },
            )
        )

    if "palette_png_sequence" in config.families:
        path = output_dir / PALETTE_PNG_MEMBER
        try:
            decoded, details = _encode_palette_png_sequence(source_masks, path)
            candidates.append(
                _candidate_record(
                    name="palette_png_lossless_image_sequence",
                    family="palette_png_sequence",
                    path=path,
                    archive_member=PALETTE_PNG_MEMBER,
                    source_masks=source_masks,
                    decoded_masks=decoded,
                    source_member_size=source_member_size,
                    payload_format="palette_png_sequence_zip_v1",
                    exact=True,
                    diagnostic=False,
                    details={**details, "source_mask_u8_sha256": source_sha},
                )
            )
        except Exception as exc:  # Optional dependency/tooling path.
            candidates.append(_skipped_candidate("palette_png_lossless_image_sequence", "palette_png_sequence", str(exc)))

    if "av1_monochrome_reference" in config.families:
        path = output_dir / AV1_MONO_MEMBER
        try:
            decoded, details = _encode_av1_monochrome_reference(source_masks, path, config=config)
            candidates.append(
                _candidate_record(
                    name="av1_monochrome_diagnostic_reference",
                    family="av1_monochrome_reference",
                    path=path,
                    archive_member=AV1_MONO_MEMBER,
                    source_masks=source_masks,
                    decoded_masks=decoded,
                    source_member_size=source_member_size,
                    payload_format="tac_mask_codec_av1_monochrome_reference",
                    exact=False,
                    diagnostic=True,
                    details={**details, "source_mask_u8_sha256": source_sha},
                )
            )
        except Exception as exc:  # Optional ffmpeg/libsvtav1 path.
            candidates.append(_skipped_candidate("av1_monochrome_diagnostic_reference", "av1_monochrome_reference", str(exc)))

    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_planning_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "rank deterministic Alpha mask codec representation candidates by charged "
            "payload bytes and exact reconstruction or diagnostic agreement"
        ),
        "matrix_config": _config_record(config),
        "source": {
            **dict(source_meta),
            "decode": effective_decode_meta,
            "decoded_masks": _mask_stats(source_masks),
        },
        "candidates": candidates,
        "rankings": _rankings(candidates),
        "provenance": _provenance(command),
    }
    _assert_non_promotable(report)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _decode_masks_member_with_local_helper(
    data: bytes,
    member: str,
    *,
    config: MatrixConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    suffix = PurePosixPath(member).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_member = Path(tmp_dir) / f"mask_member{suffix or '.bin'}"
        local_member.write_bytes(data)
        if suffix in {".mkv", ".mp4", ".webm", ".avi"}:
            from tac.mask_codec import decode_masks

            masks = decode_masks(local_member)
            decode_meta = {
                "decoder": "tac.mask_codec.decode_masks",
                "member_suffix": suffix,
                "decoded_frames": int(masks.shape[0]),
                "max_frames": config.max_frames,
                "truncated_by_matrix": False,
            }
        elif suffix == ".nrv":
            if config.nrv_shape is None:
                raise ValueError("decoding .nrv mask members requires --nrv-shape T,H,W")
            from tac.nerv_mask_codec import decode_nerv_codec, render_mask_argmax

            t, h, w = [int(v) for v in config.nrv_shape]
            codec = decode_nerv_codec(local_member.read_bytes())
            masks = render_mask_argmax(codec, num_frames=t, height=h, width=w, device="cpu").long()
            decode_meta = {
                "decoder": "tac.nerv_mask_codec.decode_nerv_codec + render_mask_argmax",
                "member_suffix": suffix,
                "decoded_frames": int(masks.shape[0]),
                "nrv_shape": [t, h, w],
                "max_frames": config.max_frames,
                "truncated_by_matrix": False,
            }
        else:
            raise ValueError(
                f"unsupported mask member suffix {suffix!r}; expected AV1 video or .nrv with --nrv-shape"
            )
    return _validate_masks(masks), decode_meta


def build_candidate_matrix_from_archive(
    *,
    archive: Path,
    mask_member: str,
    output_dir: Path,
    config: MatrixConfig,
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    _validate_config(config)
    member_data, source_meta = _read_archive_member(archive, mask_member)
    masks, decode_meta = _decode_masks_member_with_local_helper(member_data, mask_member, config=config)
    return _build_candidate_matrix_from_masks(
        masks=masks,
        source_meta=source_meta,
        output_dir=output_dir,
        config=config,
        command=command,
        decode_meta=decode_meta,
        force=force,
    )


def _parse_families(value: str) -> tuple[str, ...]:
    families: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        family = raw.strip()
        if not family:
            continue
        if family in seen:
            raise ValueError(f"duplicate candidate family {family!r}")
        seen.add(family)
        families.append(family)
    return tuple(families)


def _parse_nrv_shape(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--nrv-shape must be T,H,W")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--nrv-shape must contain integers") from exc
    if any(v <= 0 for v in shape):
        raise ValueError("--nrv-shape values must be positive")
    return shape  # type: ignore[return-value]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mask-member", default="masks.mkv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"maximum decoded frames for planning ({DEFAULT_MAX_FRAMES}); use --all-frames for full stream",
    )
    parser.add_argument("--all-frames", action="store_true", help="build matrix from every decoded frame")
    parser.add_argument(
        "--families",
        default="coco_rle,component_boundary_delta,transition_endpoints,palette_png_sequence,av1_monochrome_reference",
        help=(
            "comma-separated families: coco_rle, component_boundary_delta, "
            "transition_endpoints, palette_png_sequence, av1_monochrome_reference"
        ),
    )
    parser.add_argument("--compression", choices=("none", "zlib"), default="zlib")
    parser.add_argument("--zlib-level", type=int, default=9)
    parser.add_argument("--av1-crf", type=int, default=50)
    parser.add_argument("--av1-fps", type=int, default=20)
    parser.add_argument("--nrv-shape", default=None, help="T,H,W for decoding .nrv mask members")
    parser.add_argument("--force", action="store_true", help="overwrite existing matrix artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = MatrixConfig(
        max_frames=None if args.all_frames else args.max_frames,
        families=_parse_families(args.families),
        compression=args.compression,
        zlib_level=args.zlib_level,
        av1_crf=args.av1_crf,
        av1_fps=args.av1_fps,
        nrv_shape=_parse_nrv_shape(args.nrv_shape),
    )
    report = build_candidate_matrix_from_archive(
        archive=args.archive,
        mask_member=args.mask_member,
        output_dir=args.output_dir,
        config=config,
        command=list(sys.argv if argv is None else [sys.argv[0], *argv]),
        force=bool(args.force),
    )
    manifest_path = args.output_dir / MANIFEST_NAME
    exact_ranked = report["rankings"]["exact_reconstruction_by_bytes"]
    print(f"[alpha-mask-matrix] wrote {manifest_path}")
    print(f"[alpha-mask-matrix] exact candidates: {len(exact_ranked)}")
    if exact_ranked:
        best = exact_ranked[0]
        print(
            "[alpha-mask-matrix] best exact-by-bytes: "
            f"{best['name']} {best['size_bytes']}B sha256={best['sha256']}"
        )
    print("[alpha-mask-matrix] no score claim; exact CUDA auth eval required before promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
